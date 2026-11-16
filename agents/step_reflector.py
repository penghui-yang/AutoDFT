"""
Post-step reflection and optional plan revision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from agents.llm_client import LLMClient, TokenBudgetExceeded
from common_types.workflow_models import ExecutionContext, SkeletalPlan, StepOutcomeSummary
from config import ENABLE_LLM_REFLECTION, LLM_LEVEL
from tools.workflow_prompts import build_step_reflector_prompt, format_step_outcomes_for_prompt

logger = logging.getLogger(__name__)


class StepReflector:
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client: Optional[LLMClient] = None):
        self.config = config or {}
        self.llm_client = llm_client or LLMClient(self.config)

    def is_llm_enabled(self) -> bool:
        return (self.config.get("LLM_LEVEL", LLM_LEVEL) == "full"
                and self.config.get("ENABLE_LLM_REFLECTION", ENABLE_LLM_REFLECTION))

    def reflect(
        self,
        outcome: StepOutcomeSummary,
        context: ExecutionContext,
        next_step_index: int,
    ) -> Dict[str, Any]:
        if self.is_llm_enabled():
            try:
                system_prompt, user_prompt = build_step_reflector_prompt(
                    {
                        "outcome": asdict(outcome),
                        "current_skeletal_plan": asdict(context.current_skeletal_plan) if context.current_skeletal_plan else {},
                        "material_metadata": context.metadata,
                        "prior_outcomes": format_step_outcomes_for_prompt(
                            [asdict(o) for name, o in context.step_outcomes.items() if name != outcome.step_name]
                        ),
                        "next_step_index": next_step_index,
                    }
                )
                response_text, record = self.llm_client.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    stage="reflection",
                    temperature=0.2,
                )
                payload = json.loads(response_text)
                payload["reasoning_record"] = asdict(record)
                return payload
            except TokenBudgetExceeded:
                raise
            except Exception as exc:
                logger.warning(f"Step reflector fallback for {outcome.step_name}: {exc}")

        decision = "accept"
        if not outcome.converged and outcome.ionic_steps_count > 0:
            decision = "redo"
        return {
            "decision": decision,
            "reasoning_trace": "Rule-based fallback reflection.",
            "modified_plan": None,
            "next_step_index": next_step_index,
            "reasoning_record": {
                "stage": "reflection",
                "input_context": {"step_name": outcome.step_name},
                "output_decision": {"decision": decision},
                "reasoning_trace": "Rule-based fallback reflection.",
            },
        }
