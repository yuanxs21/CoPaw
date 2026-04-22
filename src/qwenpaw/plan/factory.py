# -*- coding: utf-8 -*-
"""Factory function for PlanNotebook initialization."""
from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.config import PlanConfig
    from agentscope.plan import PlanNotebook

logger = logging.getLogger(__name__)

_PLAN_AVAILABLE = importlib.util.find_spec("agentscope.plan") is not None


def create_plan_notebook(
    config: "PlanConfig",
) -> PlanNotebook | None:
    """Instantiate PlanNotebook from PlanConfig.

    Returns None when plan is disabled or the agentscope plan module
    is not available (avoids import-time side effects).

    Plans are stored in ephemeral memory only (cleared on restart).
    """
    if not config.enabled:
        return None

    if not _PLAN_AVAILABLE:
        logger.warning(
            "Plan mode is enabled but agentscope.plan is not available. "
            "Please upgrade AgentScope to a version that includes the "
            "plan module. Plan mode will be disabled.",
        )
        return None

    from .hints import ExtendedPlanToHint
    from .notebook import JsonSubtaskPlanNotebook

    plan_to_hint = None
    if ExtendedPlanToHint is not None:
        plan_to_hint = ExtendedPlanToHint()

    notebook = JsonSubtaskPlanNotebook(
        max_subtasks=config.max_subtasks,
        storage=None,
        plan_to_hint=plan_to_hint,
    )
    # Keep historical tool methods callable for AgentScope toolkit
    # registration.
    # They are blocked by plan tool gate and overridden in notebook to return
    # explicit "disabled" responses.
    if plan_to_hint is not None:
        plan_to_hint.bind_notebook(notebook)
    return notebook
