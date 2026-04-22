# -*- coding: utf-8 -*-
"""Session-scoped PlanNotebook lifecycle helpers.

The workspace keeps a single ``PlanNotebook`` instance. Plan snapshots are
stored in an in-process ephemeral dictionary keyed by ``session_id``:
session-isolated, but cleared on process restart.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from .broadcast import broadcast_plan_update
from .hints import set_plan_gate
from .schemas import plan_to_response

logger = logging.getLogger(__name__)

_BOUND_SESSION_ATTR = "_qwenpaw_plan_bound_session_id"
_plan_state_store: dict[str, dict[str, Any] | None] = {}
_session_plan_sweep_done = False

_PLAN_TOOL_NAMES = frozenset(
    {
        "create_plan",
        "revise_current_plan",
        "update_subtask_state",
        "finish_subtask",
        "finish_plan",
        "view_subtasks",
        "view_historical_plans",
        "recover_historical_plan",
    },
)


def _entry_contains_plan_tool_trace(entry: Any) -> bool:
    """Return True when one memory entry contains plan tool use/result."""
    msg = None
    if isinstance(entry, list) and entry:
        msg = entry[0]
    elif isinstance(entry, dict):
        msg = entry
    if not isinstance(msg, dict):
        return False
    blocks = msg.get("content")
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in {"tool_use", "tool_result"}:
            continue
        name = block.get("name")
        if isinstance(name, str) and name in _PLAN_TOOL_NAMES:
            return True
    return False


def _strip_plan_tool_traces_from_memory_blob(raw: dict[str, Any]) -> bool:
    """Remove plan tool traces from persisted memory payload in place."""
    agent_blob = raw.get("agent")
    if not isinstance(agent_blob, dict):
        return False
    memory_blob = agent_blob.get("memory")
    if not isinstance(memory_blob, dict):
        return False
    content = memory_blob.get("content")
    if not isinstance(content, list):
        return False
    filtered: list[Any] = []
    changed = False
    for entry in content:
        if _entry_contains_plan_tool_trace(entry):
            changed = True
            continue
        filtered.append(entry)
    if changed:
        memory_blob["content"] = filtered
    return changed


def _strip_plan_payload_from_state_blob(raw: dict[str, Any]) -> bool:
    """Remove serialized plan payload from session JSON blob in place."""
    changed = False
    if raw.get("plan_notebook") is not None:
        raw["plan_notebook"] = None
        changed = True

    agent_blob = raw.get("agent")
    if (
        isinstance(agent_blob, dict)
        and agent_blob.get("plan_notebook") is not None
    ):
        agent_blob["plan_notebook"] = None
        changed = True
    if _strip_plan_tool_traces_from_memory_blob(raw):
        changed = True
    return changed


def _sanitize_session_json_file(path: Path) -> bool:
    """Best-effort: strip plan payload from a single session JSON file."""
    try:
        raw_text = path.read_text(encoding="utf-8")
        blob = json.loads(raw_text)
    except Exception:  # pylint: disable=broad-except
        return False
    if not isinstance(blob, dict):
        return False
    if not _strip_plan_payload_from_state_blob(blob):
        return False
    try:
        path.write_text(
            json.dumps(blob, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:  # pylint: disable=broad-except
        logger.warning("Failed to rewrite sanitized session file: %s", path)
        return False
    return True


def _sanitize_session_json_payloads_best_effort(
    session,
    *,
    session_id: str = "",
    user_id: str = "",
    sweep_all: bool = False,
) -> None:
    """Strip leaked plan payload from session JSON files.

    AgentScope serializes ``agent.plan_notebook`` inside
    ``agent.state_dict()``.
    New sessions can read these files via tools; sanitize plan payload to avoid
    cross-session plan resurrection.
    """
    if session is None:
        return
    if sweep_all:
        save_dir = getattr(session, "save_dir", "")
        if not isinstance(save_dir, str) or not save_dir:
            return
        dir_path = Path(save_dir)
        if not dir_path.exists():
            return
        changed = 0
        for p in dir_path.glob("*.json"):
            if _sanitize_session_json_file(p):
                changed += 1
        if changed:
            logger.info(
                "Sanitized leaked plan payload from %d session file(s)",
                changed,
            )
        return

    get_path = getattr(session, "_get_save_path", None)
    if not callable(get_path):
        return
    try:
        target = Path(get_path(session_id, user_id=user_id))
    except Exception:  # pylint: disable=broad-except
        return
    if target.exists() and _sanitize_session_json_file(target):
        logger.debug("Sanitized leaked plan payload in %s", target)


def _purge_legacy_plan_files_best_effort(  # pylint: disable=too-many-branches
    session,
) -> None:
    """Best-effort delete legacy file-based plan artifacts.

    Plan lifecycle is session-memory-only. Remove stale ``plans/*.json`` and
    temporary ``*.tmp`` files so new sessions cannot read historical plans
    from disk via generic file tools.
    """
    if session is None:
        return
    save_dir = getattr(session, "save_dir", "")
    if not isinstance(save_dir, str) or not save_dir:
        return
    sessions_dir = Path(save_dir)
    workspace_dir = (
        sessions_dir.parent
        if sessions_dir.name == "sessions"
        else sessions_dir
    )
    candidates: list[Path] = [workspace_dir / "plans"]
    workspaces_root = workspace_dir.parent
    if workspaces_root.name == "workspaces" and workspaces_root.is_dir():
        try:
            for child in workspaces_root.iterdir():
                if child.is_dir():
                    candidates.append(child / "plans")
        except Exception:  # pylint: disable=broad-except
            logger.debug(
                "Failed to enumerate workspaces for legacy plan cleanup",
                exc_info=True,
            )
    removed = 0
    for plans_dir in candidates:
        if not plans_dir.exists() or not plans_dir.is_dir():
            continue
        for p in plans_dir.rglob("*.json"):
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:  # pylint: disable=broad-except
                logger.debug("Failed to delete legacy plan file: %s", p)
        for p in plans_dir.rglob("*.tmp"):
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:  # pylint: disable=broad-except
                logger.debug("Failed to delete legacy temp plan file: %s", p)
    if removed:
        logger.info("Removed %d legacy plan artifact file(s)", removed)


def _get_bound_session_id(nb: Any) -> str:
    """Read the notebook's bound chat session id (best-effort)."""
    if nb is None:
        return ""
    raw = getattr(nb, _BOUND_SESSION_ATTR, "")
    return raw if isinstance(raw, str) else ""


def _bind_session_id(nb: Any, session_id: str) -> None:
    """Bind in-memory notebook to a chat session id."""
    if nb is None:
        return
    setattr(nb, _BOUND_SESSION_ATTR, (session_id or "").strip())


def has_plan_snapshot(session_id: str) -> bool:
    """Return whether *session_id* has a meaningful plan snapshot in memory.

    ``None`` (or any non-dict placeholder) is treated as *no snapshot*.
    Additionally, a dict payload whose ``current_plan`` is missing / ``null``
    is also treated as *no active-plan snapshot*.
    """
    sid = (session_id or "").strip()
    if not sid:
        return False
    payload = _plan_state_store.get(sid)
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("current_plan"), dict)


def _repeat_guard_reset(nb: Any) -> None:
    if nb is None:
        return
    if hasattr(nb, "_plan_repeat_fingerprint"):
        nb._plan_repeat_fingerprint = None  # pylint: disable=protected-access
    if hasattr(nb, "_plan_repeat_count"):
        nb._plan_repeat_count = 0  # pylint: disable=protected-access
    if hasattr(nb, "_plan_state_repeat_fingerprint"):
        # pylint: disable-next=protected-access
        nb._plan_state_repeat_fingerprint = None
    if hasattr(nb, "_plan_state_repeat_count"):
        nb._plan_state_repeat_count = 0  # pylint: disable=protected-access
    if hasattr(nb, "_plan_repeat_force_finish"):
        # pylint: disable-next=protected-access
        nb._plan_repeat_force_finish = False
    if hasattr(nb, "_plan_repeat_force_tool"):
        nb._plan_repeat_force_tool = ""  # pylint: disable=protected-access
    if hasattr(nb, "_plan_repeat_force_subtask_idx"):
        # pylint: disable-next=protected-access
        nb._plan_repeat_force_subtask_idx = -1


async def persist_plan_notebook_to_session(
    *,
    session,
    plan_notebook,
    session_id: str,
    user_id: str,
) -> None:
    """Write the in-memory notebook to the ephemeral session plan store.

    Also explicitly clears ``plan_notebook`` in the session JSON to prevent
    ``agent.state_dict()`` from leaking plan details into disk files that
    can be read by new sessions via ``read_file`` / ``memory_search``.
    """
    if plan_notebook is None or not session_id:
        return
    sid = (session_id or "").strip()
    if not sid:
        return
    # Use internal state accessor to bypass public state_dict() that returns
    # empty dict (preventing agent auto-serialization leaks).
    internal_fn = getattr(plan_notebook, "_internal_state_dict", None)
    if callable(internal_fn):
        try:
            payload = internal_fn()
        except Exception:  # pylint: disable=broad-except
            logger.warning("_internal_state_dict failed", exc_info=True)
            return
    else:
        # Fallback: directly access current_plan if no internal accessor.
        payload = {}
    if getattr(plan_notebook, "current_plan", None) is None:
        payload = None
    elif isinstance(payload, dict):
        payload = copy.deepcopy(payload)
    if payload is None:
        _plan_state_store.pop(sid, None)
    else:
        _plan_state_store[sid] = payload
    _bind_session_id(plan_notebook, sid)

    # Overwrite the ``plan_notebook`` key in session JSON with ``null``
    # so that ``agent.state_dict()`` auto-serialization does not leak plan
    # details into disk files readable by new sessions.
    if session is not None:
        try:
            await session.update_session_state(
                session_id=session_id,
                key="plan_notebook",
                value=None,
                user_id=user_id,
            )
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to clear plan_notebook in session JSON",
                exc_info=True,
            )
        _sanitize_session_json_payloads_best_effort(
            session,
            session_id=session_id,
            user_id=user_id,
            sweep_all=False,
        )
        if payload is None:
            # Plan is completed/abandoned/cancelled for this session:
            # proactively remove legacy historical plan artifacts from disk.
            _sanitize_session_json_payloads_best_effort(
                session,
                sweep_all=True,
            )
            _purge_legacy_plan_files_best_effort(session)


def hydrate_plan_from_store(
    *,
    session_id: str,
    plan_notebook,
) -> bool:
    """Load one session's plan snapshot from ephemeral memory store."""
    if plan_notebook is None or not session_id:
        return False
    sid = (session_id or "").strip()
    if not sid or sid not in _plan_state_store:
        return False
    load_fn = getattr(plan_notebook, "load_state_dict", None)
    if not callable(load_fn):
        return False
    payload = _plan_state_store.get(sid)
    state_dict = copy.deepcopy(payload) if isinstance(payload, dict) else {}
    try:
        load_fn(state_dict)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "hydrate_plan_from_store failed for %s",
            sid,
            exc_info=True,
        )
        return False
    _bind_session_id(plan_notebook, sid)
    return True


async def reset_plan_notebook_for_session_switch(
    plan_notebook,
    *,
    agent_id: str,
    outcome: str = "Session switched",
    preserve_foreign_session_plan: bool = False,
) -> None:
    """Reset shared notebook state when switching sessions/contexts.

    By default this preserves historical behavior and abandons the active plan
    through ``finish_plan``.

    When ``preserve_foreign_session_plan`` is True, only in-memory singleton
    state is cleared (no finish/abandon). This is used by cross-session
    isolation paths where the target session has no snapshot and we must not
    mutate another session's persisted active plan.
    """
    if plan_notebook is None:
        return

    if preserve_foreign_session_plan:
        logger.debug(
            "reset_plan_notebook_for_session_switch: clear in-memory state "
            "without finishing foreign session plan (%s)",
            outcome,
        )
        # Session switch isolation: drop only transient in-memory plan state.
        setattr(plan_notebook, "current_plan", None)
        if hasattr(plan_notebook, "_plan_just_mutated"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_just_mutated = False
        if hasattr(plan_notebook, "_plan_panel_revised_pending"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_panel_revised_pending = False
        if hasattr(plan_notebook, "_plan_recently_finished"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_recently_finished = False
    else:
        cur = getattr(plan_notebook, "current_plan", None)
        if cur is not None:
            try:
                await plan_notebook.finish_plan(
                    state="abandoned",
                    outcome=outcome,
                )
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "finish_plan while resetting notebook failed; "
                    "forcing clear",
                    exc_info=True,
                )
                setattr(plan_notebook, "current_plan", None)
        else:
            setattr(plan_notebook, "current_plan", None)

    set_plan_gate(plan_notebook, False)
    if hasattr(plan_notebook, "_plan_needs_reconfirmation"):
        # pylint: disable-next=protected-access
        plan_notebook._plan_needs_reconfirmation = False
    _repeat_guard_reset(plan_notebook)

    if not preserve_foreign_session_plan:
        # Only broadcast the reset when we've genuinely finished/abandoned a
        # plan that belonged to the current session (/clear, /stop).
        # Do NOT broadcast when merely clearing transient in-memory state for
        # cross-session isolation (would incorrectly push "null" to active SSE
        # clients watching a different session's plan).
        try:
            broadcast_plan_update(agent_id, None)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "broadcast after notebook reset failed",
                exc_info=True,
            )


# pylint: disable=too-many-branches
async def clear_plan_notebook_if_session_has_no_snapshot(
    *,
    session,
    plan_notebook,
    session_id: str,
    agent_id: str,
) -> None:
    """Drop stale in-memory plan when the session has no in-memory snapshot."""
    global _session_plan_sweep_done
    if plan_notebook is None or not session_id:
        return
    # Preserve explicit /plan gate in this request turn. Runner sets the gate
    # before session hydration for ``/plan <description>``; session isolation
    # resets below must not silently clear it, or non-create tools can run.
    preserve_gate = bool(
        getattr(plan_notebook, "_plan_tool_gate", False)
        and getattr(plan_notebook, "_plan_tool_gate_preserve_once", False),
    )
    if not _session_plan_sweep_done:
        _sanitize_session_json_payloads_best_effort(
            session,
            sweep_all=True,
        )
        _purge_legacy_plan_files_best_effort(session)
        _session_plan_sweep_done = True
    sid = (session_id or "").strip()
    if has_plan_snapshot(sid):
        if hasattr(plan_notebook, "_plan_fresh_session_no_plan"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_fresh_session_no_plan = False
        if hasattr(plan_notebook, "_plan_fresh_session_block_auto_continue"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_fresh_session_block_auto_continue = False
        if hasattr(plan_notebook, "_plan_fresh_session_probe_guard"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_fresh_session_probe_guard = False
        _bind_session_id(plan_notebook, sid)
        return

    # During an ongoing first turn, the session file may not be saved yet.
    # Keep in-memory plan only when it is already bound to this same session.
    # Never let a new session adopt an unbound plan from shared notebook.
    cur = getattr(plan_notebook, "current_plan", None)
    bound_sid = _get_bound_session_id(plan_notebook)
    if cur is not None and bound_sid == sid:
        if hasattr(plan_notebook, "_plan_fresh_session_no_plan"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_fresh_session_no_plan = False
        if hasattr(plan_notebook, "_plan_fresh_session_block_auto_continue"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_fresh_session_block_auto_continue = False
        if hasattr(plan_notebook, "_plan_fresh_session_probe_guard"):
            # pylint: disable-next=protected-access
            plan_notebook._plan_fresh_session_probe_guard = False
        _bind_session_id(plan_notebook, sid)
        return
    await reset_plan_notebook_for_session_switch(
        plan_notebook,
        agent_id=agent_id,
        preserve_foreign_session_plan=True,
    )
    if preserve_gate:
        set_plan_gate(plan_notebook, True)
        # Consume the one-shot preserve marker so the gate cannot leak to
        # unrelated future session switches.
        # pylint: disable-next=protected-access
        plan_notebook._plan_tool_gate_preserve_once = False
    if hasattr(plan_notebook, "_plan_fresh_session_no_plan"):
        # Mark a one-shot guard for newly scoped sessions with no active plan,
        # so the model does not "resume" by reading historical memory files.
        # pylint: disable-next=protected-access
        plan_notebook._plan_fresh_session_no_plan = bool(bound_sid != sid)
    if hasattr(plan_notebook, "_plan_fresh_session_block_auto_continue"):
        # Keep auto-continue off until the fresh-session guard is surfaced
        # once.
        # pylint: disable-next=protected-access
        plan_notebook._plan_fresh_session_block_auto_continue = bool(
            bound_sid != sid,
        )
    if hasattr(plan_notebook, "_plan_fresh_session_probe_guard"):
        # First turn in a fresh empty session: block probing session/memory
        # artifacts from workspace tools; lifted automatically on next turn.
        # pylint: disable-next=protected-access
        plan_notebook._plan_fresh_session_probe_guard = bool(bound_sid != sid)
    _bind_session_id(plan_notebook, sid)


def broadcast_plan_notebook_snapshot(plan_notebook, agent_id: str) -> None:
    """Notify SSE clients after ``load_state_dict`` on the shared notebook."""
    if plan_notebook is None:
        return
    try:
        cur = getattr(plan_notebook, "current_plan", None)
        if cur is None:
            broadcast_plan_update(agent_id, None)
        else:
            payload = plan_to_response(cur).model_dump()
            broadcast_plan_update(agent_id, payload)
    except Exception:  # pylint: disable=broad-except
        logger.warning("broadcast after notebook load failed", exc_info=True)
