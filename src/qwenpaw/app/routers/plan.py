# -*- coding: utf-8 -*-
"""Plan API endpoints for real-time plan visualization and management."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import StreamingResponse

from ...constant import EnvVarLoader
from ..agent_context import get_agent_for_request
from ..auth import _get_jwt_secret, has_registered_users, is_auth_enabled
from ...plan import set_plan_gate
from ...plan.session_sync import (
    has_plan_snapshot,
    hydrate_plan_from_store,
    persist_plan_notebook_to_session,
)
from ...plan.schemas import (
    FinishPlanRequest,
    PlanConfigUpdateRequest,
    PlanStateResponse,
    RevisePlanRequest,
    plan_to_response,
)
from ...plan.broadcast import (
    plan_sse_scope,
    plan_sse_scope_key,
    register_sse_client,
    unregister_sse_client,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plan", tags=["plan"])

# Short-lived HMAC-signed tickets for plan SSE (EventSource cannot send
# Authorization headers; avoids putting long-lived JWTs in query strings).
# Stateless tickets work across multiple workers; set
# ``QWENPAW_PLAN_SSE_SIGNING_KEY`` when workers do not share the same
# ``jwt_secret`` on disk.
_SSE_TICKET_TTL_SECONDS = 60


def _sse_ticket_signing_key() -> bytes:
    custom = EnvVarLoader.get_str("QWENPAW_PLAN_SSE_SIGNING_KEY", "").strip()
    if custom:
        return custom.encode("utf-8")
    return _get_jwt_secret().encode("utf-8")


def _issue_sse_ticket(
    agent_id: str,
    *,
    scope_key: str | None = None,
) -> str:
    payload = {"a": agent_id, "iat": time.time()}
    if scope_key:
        payload["s"] = scope_key
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii")
    body_b64 = encoded.rstrip("=")
    sig = hmac.new(
        _sse_ticket_signing_key(),
        body_b64.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{body_b64}.{sig}"


def _decode_sse_ticket_payload(raw: str) -> dict | None:
    """Decode and verify SSE ticket payload."""
    try:
        parts = raw.split(".", 1)
        if len(parts) != 2:
            return None
        body_b64, sig = parts
        expected = hmac.new(
            _sse_ticket_signing_key(),
            body_b64.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return None
        pad = "=" * (-len(body_b64) % 4)
        decoded = base64.urlsafe_b64decode(body_b64 + pad)
        payload = json.loads(decoded.decode("utf-8"))
        iat = float(payload.get("iat", 0))
        if time.time() - iat > _SSE_TICKET_TTL_SECONDS:
            return None
        return payload if isinstance(payload, dict) else None
    except Exception:  # pylint: disable=broad-except
        return None


def _consume_sse_ticket(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Return (agent_id, scope_key) for a valid ticket, else (None, None)."""
    payload = _decode_sse_ticket_payload(raw)
    if payload is None:
        return None, None

    aid = payload.get("a")
    if not isinstance(aid, str) or not aid.strip():
        return None, None

    sk_raw = payload.get("s")
    if sk_raw is not None and not isinstance(sk_raw, str):
        return None, None

    scope_key = None
    if isinstance(sk_raw, str):
        scope_key = sk_raw.strip() or None
    return aid, scope_key


async def _get_plan_notebook(request: Request):
    """Resolve the PlanNotebook for the current request's agent."""
    workspace = await get_agent_for_request(request)
    nb = workspace.plan_notebook
    if nb is None:
        raise HTTPException(
            status_code=404,
            detail="Plan mode is not enabled for this agent",
        )
    return nb, workspace


def _session_scope_from_request(request: Request) -> tuple[str, str]:
    """Return session_id and user_id from headers or query (console chat)."""
    session_id = (
        request.headers.get("X-Session-Id")
        or request.query_params.get("session_id")
        or ""
    ).strip()
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        user_id = request.query_params.get("user_id") or ""
    user_id = user_id.strip()
    return session_id, user_id


def _channel_from_request(request: Request) -> str:
    """Channel id for plan SSE scope (defaults to ``console``)."""
    return (
        request.headers.get("X-Channel")
        or request.query_params.get("channel")
        or "console"
    ).strip() or "console"


async def _hydrate_plan_notebook_from_session(
    request: Request,
    workspace,
    plan_notebook,
) -> None:
    """Load ``plan_notebook`` from ephemeral session plan store."""
    _ = workspace
    session_id, _ = _session_scope_from_request(request)
    if not session_id or plan_notebook is None:
        return
    hydrate_plan_from_store(
        session_id=session_id,
        plan_notebook=plan_notebook,
    )


async def _persist_plan_notebook_to_session(
    request: Request,
    workspace,
    plan_notebook,
) -> None:
    """Write the in-memory notebook to the chat session JSON immediately."""
    session_id, user_id = _session_scope_from_request(request)
    runner = getattr(workspace, "runner", None)
    session = getattr(runner, "session", None) if runner is not None else None
    try:
        await persist_plan_notebook_to_session(
            session=session,
            plan_notebook=plan_notebook,
            session_id=session_id,
            user_id=user_id,
        )
    except Exception as e:
        logger.exception("Failed to persist plan_notebook via HTTP")
        raise HTTPException(
            status_code=500,
            detail="Failed to persist plan notebook to session",
        ) from e


async def _is_task_already_cancelling(tracker, run_key: str) -> bool:
    """Return True only when the tracked task is already being cancelled.

    Uses Python 3.11+ ``Task.cancelling()``; on older runtimes returns
    ``False`` so callers fall back to an explicit cancel. Read under the
    tracker lock to keep state inspection consistent with mutations.
    """
    try:
        async with tracker.lock:
            state = getattr(tracker, "_runs", {}).get(run_key)
            if state is None or state.task.done():
                return False
            cancelling_fn = getattr(state.task, "cancelling", None)
            if not callable(cancelling_fn):
                return False
            try:
                return cancelling_fn() > 0
            except Exception:  # pylint: disable=broad-except
                return False
    except Exception:  # pylint: disable=broad-except
        return False


async def _stop_session_task_if_running(
    workspace,
    *,
    channel: str,
    session_id: str,
) -> bool:
    """Best-effort cancel + queue clear for panel-driven abandon."""
    if not session_id:
        return False

    stopped = await _stop_session_runtime_task(
        workspace,
        channel=channel,
        session_id=session_id,
    )
    await _clear_session_pending_queue(
        workspace,
        channel=channel,
        session_id=session_id,
    )
    return stopped


async def _stop_session_runtime_task(
    workspace,
    *,
    channel: str,
    session_id: str,
) -> bool:
    """Cancel chat-bound and external runtime tasks for one session."""
    tracker = getattr(workspace, "task_tracker", None)
    chat_manager = getattr(workspace, "chat_manager", None)
    if tracker is None or chat_manager is None:
        return False

    chat_id = await _resolve_chat_id(chat_manager, session_id, channel)
    stopped = await _stop_chat_task_if_needed(tracker, chat_id)
    await _stop_scoped_external_tasks(tracker, channel, session_id)
    return stopped


async def _resolve_chat_id(chat_manager, session_id: str, channel: str):
    try:
        return await chat_manager.get_chat_id_by_session(session_id, channel)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "plan finish: failed to resolve chat_id for session stop",
            exc_info=True,
        )
        return None


async def _stop_chat_task_if_needed(tracker, chat_id: str | None) -> bool:
    if not chat_id:
        return False
    if await _is_task_already_cancelling(tracker, chat_id):
        # chat/stop is already cancelling this task; avoid double cancel
        # so AgentException emission is not suppressed.
        return True
    try:
        return await tracker.request_stop(chat_id)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "plan finish: chat_id stop failed for %s",
            chat_id,
            exc_info=True,
        )
        return False


async def _stop_scoped_external_tasks(
    tracker,
    channel: str,
    session_id: str,
) -> None:
    ext_prefix = f"ext:{channel}:{session_id}:"
    try:
        active_keys = await tracker.list_active_tasks()
    except Exception:  # pylint: disable=broad-except
        active_keys = []
    for run_key in active_keys:
        if not run_key.startswith(ext_prefix):
            continue
        try:
            await tracker.request_stop(run_key)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "plan finish: scoped external stop failed for %s",
                run_key,
                exc_info=True,
            )


async def _clear_session_pending_queue(
    workspace,
    *,
    channel: str,
    session_id: str,
) -> None:
    channel_manager = getattr(workspace, "channel_manager", None)
    if channel_manager is None:
        return
    try:
        await channel_manager.clear_queue(channel, session_id, 20)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "plan finish: clear_queue failed for %s/%s",
            channel,
            session_id,
            exc_info=True,
        )


def _bound_session_id(plan_notebook) -> str:
    raw = getattr(plan_notebook, "_qwenpaw_plan_bound_session_id", "")
    return raw if isinstance(raw, str) else ""


def _has_active_subtask(plan_notebook) -> bool:
    cur = getattr(plan_notebook, "current_plan", None)
    if cur is None:
        return False
    subtasks = getattr(cur, "subtasks", None) or []
    return any(getattr(st, "state", None) == "in_progress" for st in subtasks)


async def _prepare_current_plan_for_session(
    request: Request,
    workspace,
    plan_notebook,
    *,
    session_id: str,
    strict: bool,
) -> bool:
    bound_sid = _bound_session_id(plan_notebook)

    # Session isolation: requests without session_id cannot read plans
    # that are bound to a specific session (prevents cross-session leaks).
    if not session_id:
        if bound_sid:
            # The in-memory plan belongs to a specific session, but this
            # request has no session_id → block to prevent leak.
            if strict:
                raise HTTPException(status_code=404, detail="No active plan")
            return False
        # No session_id and no bound plan → allow (legacy compatibility)
        return True

    has_snapshot = has_plan_snapshot(session_id)

    # Cross-session isolation: if the in-memory plan belongs to a different
    # session, block read unless we have a snapshot for the current session.
    if bound_sid and bound_sid != session_id:
        if not has_snapshot:
            if strict:
                raise HTTPException(status_code=404, detail="No active plan")
            return False
        # Fall through to hydrate from snapshot

    in_memory_authoritative = (
        bool(bound_sid) and bound_sid == session_id
    ) or _has_active_subtask(plan_notebook)

    if has_snapshot and not in_memory_authoritative:
        channel = _channel_from_request(request)
        with plan_sse_scope(channel, session_id):
            await _hydrate_plan_notebook_from_session(
                request,
                workspace,
                plan_notebook,
            )
        return True

    if not has_snapshot:
        # If the notebook is already bound to this same session and has
        # an in-memory plan (first turn not persisted yet), allow read.
        # Otherwise return empty for this session without mutating the
        # shared notebook instance.
        in_memory_same_session = plan_notebook.current_plan is not None and (
            not bound_sid or bound_sid == session_id
        )
        if not in_memory_same_session:
            if strict:
                raise HTTPException(status_code=404, detail="No active plan")
            return False
    return True


@router.get(
    "/current",
    response_model=Optional[PlanStateResponse],
    summary="Get current plan state",
)
async def get_current_plan(
    request: Request,
    strict: bool = Query(
        False,
        description=(
            "When true, respond with 404 if no plan is active; "
            "default is 200 with null body for backward compatibility."
        ),
    ),
):
    """Return the current plan state, or null if no plan is active.

    Use ``strict=true`` for API clients that prefer 404 over nullable 200.

    When ``X-Session-Id`` (or query ``session_id``) is set, sync the shared
    notebook from that session file first so callers never read another
    session's in-memory plan.
    """
    workspace = await get_agent_for_request(request)
    nb = workspace.plan_notebook
    if nb is None:
        if strict:
            raise HTTPException(status_code=404, detail="No active plan")
        return None

    session_id, _ = _session_scope_from_request(request)
    visible = await _prepare_current_plan_for_session(
        request,
        workspace,
        nb,
        session_id=session_id,
        strict=strict,
    )
    if not visible:
        return None

    if nb.current_plan is None:
        if strict:
            raise HTTPException(status_code=404, detail="No active plan")
        return None
    if session_id:
        bound_sid = _bound_session_id(nb)
        if bound_sid and bound_sid != session_id:
            if strict:
                raise HTTPException(status_code=404, detail="No active plan")
            return None
    return plan_to_response(nb.current_plan)


@router.post(
    "/revise",
    response_model=PlanStateResponse,
    summary="Revise the current plan",
)
async def revise_plan(body: RevisePlanRequest, request: Request):
    """Revise the current plan by adding, revising, or deleting a subtask."""
    nb, workspace = await _get_plan_notebook(request)
    if nb.current_plan is None:
        raise HTTPException(
            status_code=400,
            detail="No active plan to revise",
        )

    from agentscope.plan import SubTask

    subtask = None
    if body.subtask is not None:
        subtask = SubTask(
            name=body.subtask.name,
            description=body.subtask.description,
            expected_outcome=body.subtask.expected_outcome,
        )
    channel = _channel_from_request(request)
    session_id, _ = _session_scope_from_request(request)
    with plan_sse_scope(channel, session_id):
        await _hydrate_plan_notebook_from_session(request, workspace, nb)
        await nb.revise_current_plan(
            subtask_idx=body.subtask_idx,
            action=body.action,
            subtask=subtask,
        )
    await _persist_plan_notebook_to_session(request, workspace, nb)
    return plan_to_response(nb.current_plan)


@router.post(
    "/finish",
    summary="Finish or abandon the current plan",
)
async def finish_plan(body: FinishPlanRequest, request: Request):
    """Finish or abandon the current plan."""
    nb, workspace = await _get_plan_notebook(request)
    if nb.current_plan is None:
        raise HTTPException(
            status_code=400,
            detail="No active plan to finish",
        )
    channel = _channel_from_request(request)
    session_id, _ = _session_scope_from_request(request)
    if body.state == "abandoned":
        await _stop_session_task_if_running(
            workspace,
            channel=channel,
            session_id=session_id,
        )
    with plan_sse_scope(channel, session_id):
        await _hydrate_plan_notebook_from_session(request, workspace, nb)
        await nb.finish_plan(state=body.state, outcome=body.outcome)
    set_plan_gate(nb, False)
    await _persist_plan_notebook_to_session(request, workspace, nb)
    return {"success": True}


@router.post(
    "/stream/ticket",
    summary="Issue a short-lived SSE ticket for plan stream",
)
async def issue_plan_stream_ticket(request: Request):
    """Return a short-lived signed ticket for ``GET /plan/stream``.

    Call this with a normal ``Authorization: Bearer`` header; the ticket
    is bound to the resolved agent and must be used within
    ``_SSE_TICKET_TTL_SECONDS``. Signing is stateless so multiple workers
    can validate the same ticket.
    """
    workspace = await get_agent_for_request(request)
    channel = _channel_from_request(request)
    session_id, _ = _session_scope_from_request(request)
    scope_key = plan_sse_scope_key(channel, session_id)
    ticket = _issue_sse_ticket(
        workspace.agent_id,
        scope_key=scope_key,
    )
    return {"ticket": ticket}


@router.get(
    "/stream",
    summary="SSE stream for real-time plan updates",
)
async def stream_plan_updates(request: Request):
    """Open an SSE connection that emits plan_update events."""
    auth_required = is_auth_enabled() and has_registered_users()
    ticket_raw = request.query_params.get("ticket")
    if auth_required:
        if not ticket_raw:
            raise HTTPException(
                status_code=401,
                detail="Missing SSE ticket; POST /plan/stream/ticket first",
            )
        bound_agent_id, ticket_scope = _consume_sse_ticket(ticket_raw)
        if bound_agent_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired SSE ticket",
            )
        workspace = await get_agent_for_request(
            request,
            agent_id=bound_agent_id,
        )
    else:
        workspace = await get_agent_for_request(request)
        ticket_scope = None
    agent_id = workspace.agent_id

    if auth_required:
        sse_scope = ticket_scope
    else:
        channel = _channel_from_request(request)
        session_id, _ = _session_scope_from_request(request)
        sse_scope = plan_sse_scope_key(channel, session_id)

    q = register_sse_client(agent_id, scope_key=sse_scope)

    async def _event_generator():
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=60)
                    data = json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    yield f"event: plan_update\ndata: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

                if await request.is_disconnected():
                    break
        except asyncio.CancelledError:
            pass
        finally:
            unregister_sse_client(agent_id, q, scope_key=sse_scope)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/config",
    summary="Get plan configuration",
)
async def get_plan_config(request: Request):
    """Get the plan configuration for the current agent."""
    workspace = await get_agent_for_request(request)
    config = workspace.config
    return config.plan.model_dump()


@router.put(
    "/config",
    summary="Update plan configuration",
)
async def update_plan_config(
    body: PlanConfigUpdateRequest,
    request: Request,
):
    """Update the plan configuration and activate/deactivate
    the PlanNotebook dynamically (no restart required)."""
    from ...config.config import (
        PlanConfig,
        load_agent_config,
        save_agent_config,
    )

    workspace = await get_agent_for_request(request)
    agent_id = workspace.agent_id

    agent_config = load_agent_config(agent_id)
    was_enabled = agent_config.plan.enabled
    agent_config.plan = PlanConfig(**body.model_dump())
    save_agent_config(agent_id, agent_config)

    if body.enabled and not was_enabled:
        await workspace.activate_plan_notebook()
    elif not body.enabled and was_enabled:
        await workspace.deactivate_plan_notebook()
        # Drop saved plan snapshot for the current chat session so re-enable
        # does not resurrect a stale plan from disk.
        session_id, user_id = _session_scope_from_request(request)
        runner = getattr(workspace, "runner", None)
        sess = getattr(runner, "session", None) if runner is not None else None
        if sess is not None and session_id:
            try:
                await sess.update_session_state(
                    session_id=session_id,
                    key="plan_notebook",
                    value=None,
                    user_id=user_id,
                )
            except Exception:
                logger.warning(
                    "Could not clear plan_notebook in session file on disable",
                    exc_info=True,
                )

    return agent_config.plan.model_dump()


@router.post(
    "/confirm",
    summary="Confirm and start executing the current plan",
)
async def confirm_plan(request: Request):
    """Mark the first todo subtask as in_progress so the agent
    begins execution on the next user message."""
    nb, workspace = await _get_plan_notebook(request)
    if nb.current_plan is None:
        raise HTTPException(
            status_code=400,
            detail="No active plan to confirm",
        )

    channel = _channel_from_request(request)
    session_id, _ = _session_scope_from_request(request)
    with plan_sse_scope(channel, session_id):
        await _hydrate_plan_notebook_from_session(request, workspace, nb)

        plan = nb.current_plan
        for idx, st in enumerate(plan.subtasks):
            if st.state == "todo":
                await nb.update_subtask_state(idx, "in_progress")
                if hasattr(nb, "_plan_needs_reconfirmation"):
                    # pylint: disable-next=protected-access
                    nb._plan_needs_reconfirmation = False
                await _persist_plan_notebook_to_session(
                    request,
                    workspace,
                    nb,
                )
                return {
                    "confirmed": True,
                    "started_subtask_idx": idx,
                }

        if hasattr(nb, "_plan_needs_reconfirmation"):
            # pylint: disable-next=protected-access
            nb._plan_needs_reconfirmation = False
    await _persist_plan_notebook_to_session(request, workspace, nb)
    return {"confirmed": True, "started_subtask_idx": None}
