"""
LLM-assisted recovery built on top of the rule-based recovery handler.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

from agents.llm_client import LLMClient, TokenBudgetExceeded
from agents.rule_monitor import FailureReason, SuggestedFix
from agents.rule_recovery_handler import RuleRecoveryHandler
from common_types.workflow_models import ExecutionContext, StepExecutionResult, StepParameters
from config import ENABLE_LLM_RECOVERY, LLM_LEVEL
from tools.file_parsers import extract_oszicar_tail
from tools.workflow_prompts import build_reasoning_recovery_prompt

logger = logging.getLogger(__name__)


class ReasoningRecovery:
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client: Optional[LLMClient] = None):
        self.config = config or {}
        self.llm_client = llm_client or LLMClient(self.config)

    def recover(
        self,
        step_parameters: StepParameters,
        execution_result: StepExecutionResult,
        context: ExecutionContext,
    ) -> StepParameters:
        if (self.config.get("LLM_LEVEL", LLM_LEVEL) == "full"
                and self.config.get("ENABLE_LLM_RECOVERY", ENABLE_LLM_RECOVERY)):
            recovery_context = {
                "material_id": context.material_id,
                "step_parameters": asdict(step_parameters),
                "execution_result": asdict(execution_result),
                "step_outcomes": {k: asdict(v) for k, v in context.step_outcomes.items()},
            }
            # Enrich with structured VASP error info so the LLM can diagnose
            metrics = execution_result.metrics or {}
            if metrics.get("failure_reason"):
                recovery_context["failure_reason"] = metrics["failure_reason"]
            if metrics.get("suggested_fix"):
                recovery_context["suggested_fix"] = metrics["suggested_fix"]
            if metrics.get("vasp_error_detail"):
                recovery_context["vasp_error_detail"] = metrics["vasp_error_detail"]
            # Append OUTCAR tail so LLM sees raw VASP output even if structured
            # diagnosis missed something
            outcar_path = os.path.join(execution_result.output_dir or "", "OUTCAR")
            if os.path.isfile(outcar_path):
                try:
                    with open(outcar_path, "r") as f:
                        lines = f.readlines()
                    recovery_context["outcar_tail"] = "".join(lines[-50:])
                except Exception:
                    pass
            output_dir = execution_result.output_dir or ""
            recovery_context["oszicar_tail"] = extract_oszicar_tail(os.path.join(output_dir, "OSZICAR"), n=30)
            previous_attempts = sorted(
                d for d in (os.listdir(output_dir) if os.path.isdir(output_dir) else [])
                if d.startswith("attempt_")
            )
            recovery_context["previous_attempts"] = previous_attempts
            recovery_context["attempt_number"] = len(previous_attempts) + 1
            last_error: Optional[str] = None
            for attempt in range(3):
                try:
                    system_prompt, user_prompt = build_reasoning_recovery_prompt(
                        recovery_context, validation_error=last_error
                    )
                    response_text, record = self.llm_client.call(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        stage="recovery",
                        temperature=0.1,
                    )
                    payload = json.loads(response_text)
                    updated = payload.get("updated_parameters", {})

                    # Validate: exactly one of incar / incar_overrides must be provided
                    has_incar = updated.get("incar") is not None
                    has_overrides = bool(updated.get("incar_overrides"))
                    if has_incar and has_overrides:
                        last_error = "Both 'incar' and 'incar_overrides' were provided. Provide exactly one."
                        logger.warning(f"LLM recovery attempt {attempt+1}: {last_error}")
                        continue
                    if not has_incar and not has_overrides:
                        last_error = "Neither 'incar' nor 'incar_overrides' was provided (both null/missing). Provide exactly one."
                        logger.warning(f"LLM recovery attempt {attempt+1}: {last_error}")
                        continue

                    # Resolve incar
                    if has_incar:
                        new_incar = updated["incar"]
                    else:
                        new_incar = {**(step_parameters.incar or {}), **updated["incar_overrides"]}

                    # Resolve kpoints (same mutual-exclusion logic, but non-fatal)
                    if updated.get("kpoints") is not None:
                        new_kpoints = updated["kpoints"]
                    elif updated.get("kpoints_overrides"):
                        new_kpoints = updated["kpoints_overrides"]
                    else:
                        new_kpoints = step_parameters.kpoints

                    self._archive_failed_run(execution_result)
                    # Build metadata without the old reasoning_record so it doesn't overwrite ours
                    base_meta = {k: v for k, v in step_parameters.metadata.items() if k != "reasoning_record"}
                    return StepParameters(
                        incar=new_incar,
                        kpoints=new_kpoints,
                        additional_files=updated.get("additional_files", step_parameters.additional_files) or step_parameters.additional_files,
                        vasp_version=updated.get("vasp_version", step_parameters.vasp_version),
                        reasoning_trace=payload.get("reasoning_trace", ""),
                        confidence=updated.get("confidence", step_parameters.confidence),
                        fallback_parameters=updated.get("fallback_parameters", step_parameters.fallback_parameters),
                        metadata=self._mark_retry({"reasoning_record": asdict(record), **base_meta}),
                    )
                except TokenBudgetExceeded:
                    raise
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(f"LLM recovery attempt {attempt+1} failed: {exc}")
            logger.warning("LLM recovery failed after 3 attempts, falling back to rule-based handler")

        handler = RuleRecoveryHandler(execution_result.output_dir)
        actual_reason, actual_fix = self._failure_signature(execution_result)
        if handler.can_retry():
            result = handler.prepare_retry(actual_reason, actual_fix)
            if result.success:
                # The executor regenerates INCAR from StepParameters, so carry the
                # rule-based INCAR changes over explicitly.
                incar = dict(step_parameters.incar or {})
                for mod in result.modifications_applied:
                    incar.pop(mod.parameter, None)
                modified = handler._parse_incar(os.path.join(execution_result.output_dir, "INCAR"))
                for mod in result.modifications_applied:
                    if mod.parameter in modified:
                        incar[mod.parameter] = modified[mod.parameter]
                incar["ISTART"] = 0
                return StepParameters(
                    incar=incar,
                    kpoints=step_parameters.kpoints,
                    additional_files=step_parameters.additional_files,
                    vasp_version=step_parameters.vasp_version,
                    reasoning_trace=f"Rule-based recovery: {actual_reason.value} -> {actual_fix.value}.",
                    confidence=0.3,
                    metadata=self._mark_retry({
                        **step_parameters.metadata,
                        "reasoning_record": {
                            "stage": "recovery",
                            "input_context": {"output_dir": execution_result.output_dir,
                                              "failure_reason": actual_reason.value,
                                              "suggested_fix": actual_fix.value},
                            "output_decision": {"rule_based": True,
                                                "failure_reason": actual_reason.value,
                                                "suggested_fix": actual_fix.value,
                                                "incar_changes": [m.parameter for m in result.modifications_applied]},
                            "reasoning_trace": f"Rule-based recovery applied: {actual_reason.value} -> {actual_fix.value}.",
                        },
                    }),
                )
        return step_parameters

    @staticmethod
    def _mark_retry(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Tell the executor to initialise the retry from the archived failed run."""
        additional = dict(metadata.get("additional_params", {}))
        additional["restart_from_archive"] = True
        return {**metadata, "additional_params": additional}

    @staticmethod
    def _failure_signature(execution_result: StepExecutionResult):
        """Return the (FailureReason, SuggestedFix) reported by the monitor, if any."""
        metrics = execution_result.metrics or {}
        raw_reason = metrics.get("failure_reason") or "UNKNOWN_ERROR"
        raw_fix = metrics.get("suggested_fix") or "FRESH_START"
        try:
            reason = FailureReason[raw_reason] if isinstance(raw_reason, str) else raw_reason
            fix = SuggestedFix[raw_fix] if isinstance(raw_fix, str) else raw_fix
        except KeyError:
            reason, fix = FailureReason.UNKNOWN_ERROR, SuggestedFix.FRESH_START
        return reason, fix

    @staticmethod
    def _archive_failed_run(execution_result: StepExecutionResult) -> None:
        """Move the failed run's files into an attempt_* archive before the retry."""
        if not execution_result.output_dir or not os.path.isdir(execution_result.output_dir):
            return
        try:
            reason, _ = ReasoningRecovery._failure_signature(execution_result)
            RuleRecoveryHandler(execution_result.output_dir)._archive_failed_calculation(reason)
        except Exception as exc:
            logger.warning(f"Could not archive failed run in {execution_result.output_dir}: {exc}")
