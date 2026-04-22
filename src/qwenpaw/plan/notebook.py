# -*- coding: utf-8 -*-
"""PlanNotebook subclass: tolerate LLM tool calls that pass JSON strings."""

import json
import logging
from typing import Any, Literal

from agentscope.plan import PlanNotebook, SubTask
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from .broadcast import current_plan_scope_session_id

# Do NOT use ``from __future__ import annotations`` here: AgentScope builds
# tool JSON schema via ``inspect.signature`` + Pydantic ``create_model``;
# postponed annotations stringify ``Literal`` / ``SubTask`` and trigger
# PydanticUserError: class not fully defined (breaks all channels).

logger = logging.getLogger(__name__)

# Cap JSON string size before ``json.loads`` to limit CPU / memory from
# pathological tool arguments (defense in depth; SubTask text is small).
_MAX_SUBTASK_JSON_CHARS = 512_000
_BOUND_SESSION_ATTR = "_qwenpaw_plan_bound_session_id"


def _normalize_subtask_payload(subtask: Any) -> Any:
    """Decode JSON string payloads so ``SubTask.model_validate`` succeeds.

    Some chat models emit ``subtask`` as a serialized JSON object string
    instead of a structured argument; AgentScope only coerces ``dict``,
    not ``str``, which caused repeated validation failures and retry loops.
    """
    if subtask is None or not isinstance(subtask, str):
        return subtask
    raw = subtask.strip()
    if len(raw) > _MAX_SUBTASK_JSON_CHARS:
        logger.warning(
            "subtask JSON string exceeds %s chars; rejecting parse",
            _MAX_SUBTASK_JSON_CHARS,
        )
        return subtask
    if len(raw) < 2 or raw[0] not in "{[":
        return subtask
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug(
            "subtask string is not valid JSON; passing through unchanged",
        )
        return subtask
    return parsed


def _bind_notebook_to_scope_session(plan_notebook: Any) -> None:
    """Bind notebook to current request session from ``plan_sse_scope``."""
    sid = current_plan_scope_session_id()
    if not sid or plan_notebook is None:
        return
    setattr(plan_notebook, _BOUND_SESSION_ATTR, sid)


class JsonSubtaskPlanNotebook(PlanNotebook):
    """PlanNotebook subclass; coerce JSON-string subtask tool arguments.

    Models sometimes pass subtask as a JSON string; we decode before
    AgentScope validates.

    When the user (or agent) revises the plan while every subtask is still
    *todo*, we require another explicit confirmation before execution; see
    :attr:`_plan_needs_reconfirmation` and plan hints.
    """

    def state_dict(self) -> dict[str, Any]:
        """Return empty dict to prevent agent auto-serialization leaks.

        Plan snapshots are managed by ``session_sync._plan_state_store``;
        returning empty here stops ``agent.state_dict()`` from writing plan
        details into session JSON files readable by new sessions.
        """
        return {}

    def _internal_state_dict(self) -> dict[str, Any]:
        """Internal state dict for ``session_sync`` persist.

        This bypasses public agent-export behavior.
        """
        payload = {}
        try:
            payload = super().state_dict()
        except Exception:  # pylint: disable=broad-except
            pass
        payload["_plan_recently_finished"] = bool(
            getattr(self, "_plan_recently_finished", False),
        )
        return payload

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load notebook state and restore one-shot post-finish marker."""
        if not state_dict:
            prev_gate = bool(getattr(self, "_plan_tool_gate", False))
            # ``plan_notebook: null`` is our session-persistence sentinel for
            # "no active plan".  Do NOT forward an empty dict to AgentScope's
            # StateModule loader because it expects the full schema
            # (including ``storage``) and will raise a KeyError. Clearing here
            # keeps the shared notebook isolated per session while avoiding the
            # load_session_state schema-mismatch path that would otherwise skip
            # all state loading and leave stale in-memory plan state behind.
            self.current_plan = None
            self._plan_recently_finished = False
            self._plan_needs_reconfirmation = False
            self._plan_just_mutated = False
            self._plan_panel_revised_pending = False
            self._plan_fresh_session_no_plan = False
            self._plan_fresh_session_block_auto_continue = False
            self._plan_fresh_session_probe_guard = False
            # Empty load means no active plan for this session; clear
            # session-scoped transient guards so they do not bleed across
            # sessions on the shared notebook instance.
            setattr(self, _BOUND_SESSION_ATTR, "")
            from .hints import set_plan_gate

            # Preserve explicit /plan gate from the current request turn.
            # Query flow sets gate before session load; clearing it here causes
            # non-create tools to bypass plan creation in the same turn.
            set_plan_gate(self, prev_gate)
            if hasattr(self, "_plan_repeat_fingerprint"):
                self._plan_repeat_fingerprint = None
            if hasattr(self, "_plan_repeat_count"):
                self._plan_repeat_count = 0
            if hasattr(self, "_plan_state_repeat_fingerprint"):
                self._plan_state_repeat_fingerprint = None
            if hasattr(self, "_plan_state_repeat_count"):
                self._plan_state_repeat_count = 0
            return
        payload = dict(state_dict or {})
        marker = bool(payload.pop("_plan_recently_finished", False))
        super().load_state_dict(payload)
        self._plan_recently_finished = marker
        self._plan_fresh_session_no_plan = False
        self._plan_fresh_session_block_auto_continue = False
        self._plan_fresh_session_probe_guard = False

    async def create_plan(
        self,
        name: str,
        description: str,
        expected_outcome: str,
        subtasks: list[SubTask],
    ) -> ToolResponse:
        if isinstance(subtasks, list):
            subtasks = [
                _normalize_subtask_payload(st) if isinstance(st, str) else st
                for st in subtasks
            ]
        resp = await super().create_plan(
            name,
            description,
            expected_outcome,
            subtasks,
        )
        self._plan_needs_reconfirmation = False
        self._plan_just_mutated = True
        self._plan_recently_finished = False
        self._plan_fresh_session_no_plan = False
        self._plan_fresh_session_block_auto_continue = False
        self._plan_fresh_session_probe_guard = False
        from .hints import set_plan_gate

        # /plan gate is only for forcing create_plan. Clear immediately after
        # successful creation so it cannot bleed into subsequent turns/
        # sessions.
        set_plan_gate(self, False)
        # Fresh plan supersedes any pending panel-revised notice.
        self._plan_panel_revised_pending = False
        _bind_notebook_to_scope_session(self)
        return resp

    async def revise_current_plan(
        self,
        subtask_idx: int,
        action: Literal["add", "revise", "delete"],
        subtask: SubTask | None = None,
    ) -> ToolResponse:
        normalized = _normalize_subtask_payload(subtask)
        resp = await super().revise_current_plan(
            subtask_idx,
            action,
            normalized,
        )
        plan = self.current_plan
        if plan is not None:
            # Auto-abandon when all subtasks removed (empty plan)
            if not plan.subtasks:
                logger.info(
                    "revise_current_plan: all subtasks removed; "
                    "auto-abandoning plan",
                )
                await self.finish_plan(
                    state="abandoned",
                    outcome="All subtasks removed by user",
                )
                # Reset gate so next turn doesn't force create_plan
                from .hints import set_plan_gate

                set_plan_gate(self, False)
            elif all(st.state == "todo" for st in plan.subtasks):
                # All remaining subtasks are still todo → need reconfirmation
                self._plan_needs_reconfirmation = True
                self._plan_just_mutated = True
        self._plan_fresh_session_no_plan = False
        self._plan_fresh_session_block_auto_continue = False
        self._plan_fresh_session_probe_guard = False
        # Mid-execution edits (some subtask already in_progress / done) do
        # not satisfy the all-todo branch above, so neither flag fires for
        # them. Mark a one-shot panel-revised notice so the next reasoning
        # hint reminds the model to follow the updated plan structure
        # rather than the original user message. Consumed in
        # ``hints.ExtendedPlanToHint._pick_hint``.
        self._plan_panel_revised_pending = True
        return resp

    async def update_subtask_state(
        self,
        subtask_idx: int,
        state: str,
    ) -> ToolResponse:
        resp = await super().update_subtask_state(subtask_idx, state)
        if state == "in_progress":
            self._plan_needs_reconfirmation = False
            self._plan_fresh_session_no_plan = False
            self._plan_fresh_session_block_auto_continue = False
            self._plan_fresh_session_probe_guard = False
        return resp

    def list_tools(self):
        """List plan tools exposed to the agent (historical tools disabled)."""
        return [
            self.view_subtasks,
            self.update_subtask_state,
            self.finish_subtask,
            self.create_plan,
            self.revise_current_plan,
            self.finish_plan,
        ]

    async def view_historical_plans(self) -> ToolResponse:
        """Override AgentScope: historical plan listing is not supported."""
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        "Historical plan listing is disabled. Do not call "
                        "this tool. Plans are session-scoped; a new chat has "
                        "no prior plans to list. Continue using only the "
                        "current conversation and workspace."
                    ),
                ),
            ],
        )

    async def recover_historical_plan(self, plan_id: str) -> ToolResponse:
        """Override AgentScope: historical plan recovery is unsupported."""
        _ = plan_id
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        "Recovering historical plans is disabled. Do not call "
                        "this tool. There is no cross-session plan recovery. "
                        "If the user wants a plan, use create_plan for this "
                        "session only."
                    ),
                ),
            ],
        )

    def _get_bound_session_id(self) -> str:
        """Read the notebook's bound chat session id."""
        raw = getattr(self, "_qwenpaw_plan_bound_session_id", "")
        return raw if isinstance(raw, str) else ""

    async def finish_plan(
        self,
        state: Literal["done", "abandoned"],
        outcome: str,
    ) -> ToolResponse:
        """Finish plan and clear active state.

        Historical persistence remains disabled.
        """
        if self.current_plan is None:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="There is no plan to finish.",
                    ),
                ],
            )
        self.current_plan.finish(state, outcome)
        self.current_plan = None
        await self._trigger_plan_change_hooks()

        self._plan_needs_reconfirmation = False
        self._plan_just_mutated = False
        self._plan_recently_finished = True
        self._plan_fresh_session_no_plan = False
        self._plan_fresh_session_block_auto_continue = False
        self._plan_fresh_session_probe_guard = False
        self._plan_panel_revised_pending = False
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        "The current plan is finished "
                        f"successfully as '{state}'."
                    ),
                ),
            ],
        )
