"""
Dual-path runtime monitor.

The rule-based path runs at every check and terminates unambiguous failures
directly, with no LLM cost. The LLM path is consulted only when the rule path
reports a non-Continue status that it does not resolve itself (no output yet
is not such a status), or as a periodic fallback every LLM_MONITOR_INTERVAL
rule-based checks.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

from agents.llm_client import LLMClient, TokenBudgetExceeded
from agents.rule_monitor import CalculationStatus, RuleMonitor
from agents.step_planner import _parse_incar_file
from config import ENABLE_LLM_MONITORING, LLM_LEVEL, LLM_MONITOR_INTERVAL
from tools.file_parsers import extract_oszicar_tail, extract_outcar_summary
from tools.workflow_prompts import build_reasoning_monitor_prompt

logger = logging.getLogger(__name__)


class ReasoningMonitor:
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client: Optional[LLMClient] = None):
        self.config = config or {}
        self.rule_monitor = RuleMonitor()
        self.llm_client = llm_client or LLMClient(self.config)

    def assess_calculation(
        self,
        calc_dir: str,
        start_time: float,
        last_check_time: Optional[float] = None,
        step_parameters: Optional[Dict[str, Any]] = None,
        physical_context: Optional[Dict[str, Any]] = None,
        check_count: int = 1,
    ) -> Dict[str, Any]:
        rule_result = self.rule_monitor.assess_calculation(calc_dir, start_time, last_check_time)
        payload = rule_result.to_dict()
        payload["status"] = payload["status"].lower()

        if rule_result.status == CalculationStatus.KILL:
            payload["decision"] = "terminate"
            payload["should_kill"] = True
            return payload

        should_call_llm = (
            self.config.get("LLM_LEVEL", LLM_LEVEL) == "full"
            and self.config.get("ENABLE_LLM_MONITORING", ENABLE_LLM_MONITORING)
            and (
                payload["status"] not in {"continue", "converged", "not_started"}
                or check_count % self.config.get("LLM_MONITOR_INTERVAL", LLM_MONITOR_INTERVAL) == 0
            )
        )
        if not should_call_llm:
            payload["decision"] = "continue" if payload["status"] != "converged" else "converged"
            payload["should_kill"] = False
            return payload

        try:
            monitor_context = {
                "calc_dir": calc_dir,
                "rule_assessment": payload,
                "oszicar_tail": extract_oszicar_tail(os.path.join(calc_dir, "OSZICAR")),
                "outcar_summary": extract_outcar_summary(os.path.join(calc_dir, "OUTCAR")),
                "step_parameters": step_parameters or {"incar": _parse_incar_file(os.path.join(calc_dir, "INCAR"))},
                "physical_context": physical_context or {},
            }
            system_prompt, user_prompt = build_reasoning_monitor_prompt(monitor_context)
            response_text, record = self.llm_client.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage="monitoring",
                temperature=0.1,
            )
            llm_payload = json.loads(response_text)
            payload["llm_assessment"] = llm_payload.get("reasoning_trace", "")
            payload["decision"] = str(llm_payload.get("decision", "continue")).lower()
            payload["should_kill"] = bool(llm_payload.get("should_kill", False)) or payload["decision"] == "terminate"
            payload["reasoning_record"] = asdict(record)
            return payload
        except TokenBudgetExceeded:
            # The controller fails the workflow once the step returns; keep the
            # rule-based result for the remainder of this job.
            logger.warning("Token budget exhausted during monitoring; using rule-based result")
            payload["decision"] = "continue" if payload["status"] != "converged" else "converged"
            payload["should_kill"] = False
            return payload
        except Exception as exc:
            logger.warning(f"Reasoning monitor fallback to rule result for {calc_dir}: {exc}")
            payload["decision"] = "continue" if payload["status"] != "converged" else "converged"
            payload["should_kill"] = False
            return payload
