# -*- coding: utf-8 -*-
"""Detect repeated identical non-plan tool calls during plan execution.

When a subtask is ``in_progress``, the model sometimes re-runs the same
shell command or file read in a loop without calling ``finish_subtask``.
This guard blocks the Nth identical call and returns an error that routes
through the same path as :func:`check_plan_tool_gate` so the model must
advance the plan instead.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Block when the same fingerprint is seen this many times in a row
# (allows three identical successful runs, stops the fourth).
_REPEAT_THRESHOLD = 4

# Plan notebook tools: do not count toward repetition;
# ``finish_subtask`` clears.
_PLAN_TOOL_NAMES = frozenset(
    {
        "finish_subtask",
        "update_subtask_state",
        "finish_plan",
        "create_plan",
        "revise_current_plan",
        "view_subtasks",
    },
)

_PLAN_STATE_TRANSITION_TOOLS = frozenset(
    {
        "update_subtask_state",
        "finish_subtask",
        "finish_plan",
    },
)


def _tool_fingerprint(tool_name: str, tool_input: Any) -> str:
    """Stable short string for comparing 'same' tool invocations."""
    if isinstance(tool_input, dict):
        try:
            payload = json.dumps(
                tool_input,
                sort_keys=True,
                ensure_ascii=False,
            )
        except TypeError:
            payload = str(tool_input)
    else:
        payload = str(tool_input)
    return f"{tool_name}:{payload[:800]}"


def _reset_repeat_state(plan_notebook) -> None:
    # pylint: disable=protected-access
    plan_notebook._plan_repeat_fingerprint = None
    plan_notebook._plan_repeat_count = 0
    plan_notebook._plan_state_repeat_fingerprint = None
    plan_notebook._plan_state_repeat_count = 0
    plan_notebook._plan_repeat_force_finish = False
    plan_notebook._plan_repeat_force_tool = ""
    plan_notebook._plan_repeat_force_subtask_idx = -1


def _in_progress_subtask_idx(plan) -> int | None:
    """Return first in-progress subtask index, else ``None``."""
    for idx, st in enumerate(getattr(plan, "subtasks", []) or []):
        if getattr(st, "state", "") == "in_progress":
            return idx
    return None


# pylint: disable=too-many-branches,too-many-return-statements
def check_plan_repeat_guard(
    plan_notebook,
    tool_name: str,
    tool_input: Any,
) -> str | None:
    """If the same non-plan tool would run too many times in a row, block.

    Only active when ``current_plan`` exists and at least one subtask is
    ``in_progress``.  Resets when ``finish_subtask`` is invoked.
    """
    if plan_notebook is None:
        return None
    plan = getattr(plan_notebook, "current_plan", None)
    if plan is None:
        return None
    if not any(st.state == "in_progress" for st in plan.subtasks):
        if hasattr(plan_notebook, "_plan_repeat_force_finish"):
            _reset_repeat_state(plan_notebook)
        return None

    if tool_name in _PLAN_STATE_TRANSITION_TOOLS:
        if not hasattr(plan_notebook, "_plan_state_repeat_fingerprint"):
            _reset_repeat_state(plan_notebook)
        fp = _tool_fingerprint(tool_name, tool_input)
        state_last_fp = getattr(
            plan_notebook,
            "_plan_state_repeat_fingerprint",
            None,
        )
        # pylint: disable=protected-access
        if fp == state_last_fp:
            plan_notebook._plan_state_repeat_count = (
                int(
                    getattr(
                        plan_notebook,
                        "_plan_state_repeat_count",
                        0,
                    ),
                )
                + 1
            )
        else:
            plan_notebook._plan_state_repeat_fingerprint = fp
            plan_notebook._plan_state_repeat_count = 1

        state_count = int(
            getattr(plan_notebook, "_plan_state_repeat_count", 0),
        )
        if state_count >= _REPEAT_THRESHOLD:
            return (
                "Repeated identical plan state-transition calls detected. "
                "Do not keep calling the same "
                f"'{tool_name}' with identical inputs. "
                "Call 'view_subtasks' to refresh current states, then either "
                "advance to the correct next transition or explain to the "
                "user what is blocked."
            )
        return None

    if tool_name in _PLAN_TOOL_NAMES:
        if tool_name in _PLAN_STATE_TRANSITION_TOOLS:
            _reset_repeat_state(plan_notebook)
        return None

    if not hasattr(plan_notebook, "_plan_repeat_fingerprint"):
        _reset_repeat_state(plan_notebook)

    fp = _tool_fingerprint(tool_name, tool_input)
    last_fp = getattr(plan_notebook, "_plan_repeat_fingerprint", None)
    # pylint: disable=protected-access
    if fp == last_fp:
        plan_notebook._plan_repeat_count = (
            int(
                getattr(
                    plan_notebook,
                    "_plan_repeat_count",
                    0,
                ),
            )
            + 1
        )
    else:
        plan_notebook._plan_repeat_fingerprint = fp
        plan_notebook._plan_repeat_count = 1

    count = int(getattr(plan_notebook, "_plan_repeat_count", 0))
    if count >= _REPEAT_THRESHOLD:
        ip_idx = _in_progress_subtask_idx(plan)
        # Escalate from soft reminder to hard gate: subsequent non-plan-state
        # tools are blocked in ``check_plan_tool_gate`` until state advances.
        # pylint: disable=protected-access
        plan_notebook._plan_repeat_force_finish = True
        plan_notebook._plan_repeat_force_tool = tool_name
        plan_notebook._plan_repeat_force_subtask_idx = (
            int(ip_idx) if ip_idx is not None else -1
        )
        logger.warning(
            "Plan repeat guard: blocking identical tool=%s "
            "after %d consecutive "
            "calls (agent_id context on notebook)",
            tool_name,
            count,
        )
        return (
            "Repeated identical tool calls while a plan subtask is "
            "in_progress. Hard guard is now armed. "
            "If the last outputs already show success (e.g. tests "
            "passed, syntax "
            "OK), you MUST call 'finish_subtask' now with the current "
            "subtask_idx and a short outcome — do not run the same "
            "command again. "
            "If something failed, use a different approach or report "
            "the error in "
            "the finish_subtask outcome."
        )
    return None
