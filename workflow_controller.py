#!/usr/bin/env python3
"""
Adaptive workflow controller: hierarchical planning and closed-loop execution
(Algorithm 1 in the paper).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agents.agent_factory import WorkflowOrchestrator
from agents.llm_client import TokenBudgetExceeded
from common_types.workflow_models import ExecutionContext, ReasoningRecord, SkeletalPlan, SkeletalStep, StepExecutionResult, WorkflowReasoningLog
from config import MAX_PLAN_REVISIONS, MAX_RETRIES_PER_STEP, MAX_TOTAL_STEPS, STEP_TIMEOUT_HOURS

logger = logging.getLogger(__name__)


class WorkflowStepLimitExceeded(RuntimeError):
    """Raised when a workflow reaches MAX_TOTAL_STEPS VASP steps."""


class WorkflowController:
    def __init__(self, work_base_dir: str, config: Optional[Dict[str, Any]] = None):
        self.work_base_dir = work_base_dir
        self.config = config or {}
        self.orchestrator = WorkflowOrchestrator(work_base_dir, config)
        self.total_pending_time = 0.0
        self.total_running_time = 0.0

    def run_workflow(
        self,
        material_id: str,
        task_description: str,
        cif_path: str,
        step_timeout_hours: float = STEP_TIMEOUT_HOURS,
        enable_monitoring: bool = True,
        material_type: str = "auto",
        conservative_mode: bool = False,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        material_dir = os.path.join(self.work_base_dir, material_id)
        results_file = os.path.join(material_dir, "results.json")

        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as handle:
                    return json.load(handle)
            except Exception:
                pass

        # Per-workflow VASP timing: the controller is reused across examples.
        self.total_pending_time = 0.0
        self.total_running_time = 0.0
        self.orchestrator.reset_workflow_budget()
        try:
            return self._run_workflow(
                material_id, task_description, cif_path, step_timeout_hours,
                enable_monitoring, material_type, conservative_mode, **kwargs,
            )
        except TokenBudgetExceeded as exc:
            logger.error(f"[{material_id}] Token budget exhausted: {exc}")
            failure = self._failure_payload(material_id, "token_budget_exceeded", None, str(exc))
        except WorkflowStepLimitExceeded as exc:
            logger.error(f"[{material_id}] {exc}")
            failure = self._failure_payload(material_id, "step_limit_exceeded", None, str(exc))
        self._save_json(results_file, failure)
        return failure

    def _run_workflow(
        self,
        material_id: str,
        task_description: str,
        cif_path: str,
        step_timeout_hours: float,
        enable_monitoring: bool,
        material_type: str,
        conservative_mode: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        material_dir = os.path.join(self.work_base_dir, material_id)
        results_file = os.path.join(material_dir, "results.json")
        reasoning_file = os.path.join(material_dir, "reasoning_log.json")
        workflow_start = time.time()
        additional_context = kwargs.get("additional_context", "")

        skeletal_plan = self.orchestrator.strategic_planner.plan(
            task_description=task_description,
            material_id=material_id,
            cif_path=cif_path,
            material_type=material_type,
            additional_context=additional_context,
            conservative_mode=conservative_mode,
        )
        context = ExecutionContext(
            material_id=material_id,
            cif_path=cif_path,
            work_base_dir=self.work_base_dir,
            formula=skeletal_plan.material_formula,
            metadata={"material_type": skeletal_plan.material_type},
            reasoning_log=WorkflowReasoningLog(material_id=material_id),
            current_skeletal_plan=skeletal_plan,
        )
        os.makedirs(material_dir, exist_ok=True)
        self._append_reasoning_record(context, skeletal_plan.metadata.get("reasoning_record"))
        self._save_json(os.path.join(material_dir, "skeletal_plan.json"), asdict(skeletal_plan))
        step_index = 0
        plan_revisions = 0
        redo_counts: dict = {}   # step_index -> number of consecutive redo decisions
        final_result: Optional[Dict[str, Any]] = None
        max_retries = self.config.get("MAX_RETRIES_PER_STEP", MAX_RETRIES_PER_STEP)
        max_revisions = self.config.get("MAX_PLAN_REVISIONS", MAX_PLAN_REVISIONS)

        try:
            while step_index < len(context.current_skeletal_plan.steps):
                if len(context.vasp_completed_steps) >= self.config.get("MAX_TOTAL_STEPS", MAX_TOTAL_STEPS):
                    raise WorkflowStepLimitExceeded("Workflow aborted: maximum number of VASP steps reached")

                skeletal_step = context.current_skeletal_plan.steps[step_index]
                if not self.orchestrator.step_planner.evaluate_condition(skeletal_step, context.step_outcomes):
                    step_index += 1
                    continue

                step_parameters = self.orchestrator.step_planner.plan_step(skeletal_step, context)
                self._append_reasoning_record(context, step_parameters.metadata.get("reasoning_record"))
                workflow_step = self.orchestrator.step_planner.materialize_workflow_step(skeletal_step, step_parameters)

                # Run -> monitor -> recover, with at most `max_retries` attempts.
                attempts = 0
                while True:
                    execution_result = self.orchestrator.vasp_executor.execute(
                        step=workflow_step,
                        context=context,
                        step_index=step_index,
                        timeout_hours=step_timeout_hours,
                        enable_monitoring=enable_monitoring,
                    )
                    self._accumulate_timing(execution_result)
                    if self.orchestrator.token_budget_exhausted():
                        raise TokenBudgetExceeded("Token budget exhausted during step execution")

                    if execution_result.status == "success":
                        break

                    attempts += 1
                    if attempts >= max_retries:
                        break

                    step_parameters = self.orchestrator.reasoning_recovery.recover(
                        step_parameters=step_parameters,
                        execution_result=execution_result,
                        context=context,
                    )
                    self._append_monitor_reasoning(context, execution_result)
                    self._append_reasoning_record(context, step_parameters.metadata.get("reasoning_record"))
                    workflow_step = self.orchestrator.step_planner.materialize_workflow_step(skeletal_step, step_parameters)

                outcome = self.orchestrator.vasp_executor.generate_step_outcome_summary(
                    step=workflow_step,
                    execution_result=execution_result,
                    context=context,
                )
                self._append_monitor_reasoning(context, execution_result)

                if execution_result.status != "success":
                    # Retries exhausted: the Step Reflector may restructure the remaining
                    # plan around the unresolvable failure; otherwise the workflow fails.
                    revised = False
                    if plan_revisions < max_revisions and self.orchestrator.step_reflector.is_llm_enabled():
                        reflection = self.orchestrator.step_reflector.reflect(outcome, context, step_index + 1)
                        self._append_reasoning_record(context, reflection.get("reasoning_record"))
                        if reflection.get("decision") == "modify_plan":
                            new_index = self._apply_plan_modification(context, reflection, step_index, material_dir)
                            if new_index is not None:
                                plan_revisions += 1
                                step_index = new_index
                                revised = True
                    if revised:
                        continue
                    failure = self._build_failure_result(material_id, skeletal_step.step_name, execution_result)
                    self._save_json(results_file, failure)
                    self._save_json(reasoning_file, asdict(context.reasoning_log))
                    return failure

                context.completed_steps[workflow_step.name] = execution_result.output_dir
                if workflow_step.agent_type == "computing" and workflow_step.INCAR:
                    context.vasp_completed_steps[workflow_step.name] = execution_result.output_dir
                context.step_outputs[workflow_step.name] = execution_result.extracted_data
                context.step_outcomes[workflow_step.name] = outcome
                final_result = execution_result.extracted_data or final_result

                reflection = self.orchestrator.step_reflector.reflect(outcome, context, step_index + 1)
                self._append_reasoning_record(context, reflection.get("reasoning_record"))
                decision = reflection.get("decision", "accept")
                if decision == "redo":
                    redo_counts[step_index] = redo_counts.get(step_index, 0) + 1
                    if redo_counts[step_index] >= max_retries:
                        logger.warning(
                            f"[{material_id}] Step {skeletal_step.step_name} hit redo limit "
                            f"({redo_counts[step_index]}), advancing to next step."
                        )
                        redo_counts.pop(step_index, None)
                        step_index += 1
                    continue
                if decision == "modify_plan" and plan_revisions < max_revisions:
                    new_index = self._apply_plan_modification(context, reflection, step_index + 1, material_dir)
                    if new_index is not None:
                        plan_revisions += 1
                        step_index = new_index
                        continue

                # Accept (also when the revision budget is exhausted or the revision is invalid).
                step_index += 1
        finally:
            self._save_json(reasoning_file, asdict(context.reasoning_log))

        extracted_properties = self._post_workflow_extraction(context, material_id, **kwargs)
        total_time = time.time() - workflow_start
        result = self._build_success_result(
            material_id=material_id,
            task_description=task_description,
            total_time=total_time,
            total_steps=len(context.step_outcomes),
            extracted_properties=extracted_properties,
            final_result=final_result,
            context=context,
            skeletal_plan=context.current_skeletal_plan,
        )
        self._save_json(results_file, result)
        self._save_json(reasoning_file, asdict(context.reasoning_log))
        return result

    def _apply_plan_modification(
        self,
        context: ExecutionContext,
        reflection: Dict[str, Any],
        default_next_index: int,
        material_dir: str,
    ) -> Optional[int]:
        """
        Install a reflector-revised skeletal plan.

        Returns the index of the next step to execute, or None when the revision is
        rejected (missing plan, empty plan, or a violated dependency invariant).
        """
        modified_plan = reflection.get("modified_plan")
        if not modified_plan:
            return None
        if isinstance(modified_plan, dict):
            valid_step_fields = SkeletalStep.__dataclass_fields__
            valid_plan_fields = SkeletalPlan.__dataclass_fields__
            steps = [
                SkeletalStep(**{k: v for k, v in step.items() if k in valid_step_fields})
                for step in modified_plan.get("steps", [])
                if isinstance(step, dict) and step.get("step_name") and step.get("objective") and step.get("computation_type")
            ]
            plan_fields = {k: v for k, v in modified_plan.items() if k in valid_plan_fields and k != "steps"}
            for key in ("task_description", "material_formula", "material_type", "overall_strategy"):
                plan_fields.setdefault(key, getattr(context.current_skeletal_plan, key))
            try:
                new_plan = SkeletalPlan(steps=steps, **plan_fields)
            except TypeError as exc:
                logger.warning(f"Rejected plan revision: malformed plan ({exc})")
                return None
        else:
            new_plan = modified_plan
        if not new_plan.steps:
            logger.warning("Rejected plan revision: no valid steps")
            return None

        try:
            next_index = int(reflection.get("next_step_index", default_next_index))
        except (TypeError, ValueError):
            next_index = default_next_index
        next_index = max(0, min(next_index, len(new_plan.steps)))

        violation = self._dependency_violation(new_plan, next_index, context)
        if violation:
            logger.warning(f"Rejected plan revision: {violation}")
            return None

        context.reasoning_log.plan_revisions.append(reflection)
        context.current_skeletal_plan = new_plan
        self._save_json(os.path.join(material_dir, "skeletal_plan.json"), asdict(new_plan))
        return next_index

    @staticmethod
    def _dependency_violation(plan: SkeletalPlan, next_index: int, context: ExecutionContext) -> Optional[str]:
        """
        Check the dependency invariant: every step still to be executed must depend
        only on outputs produced by an earlier step (already completed, or scheduled
        earlier in the revised plan).
        """
        available = set(context.completed_steps)
        for step in plan.steps[:next_index]:
            available.add(step.step_name)
        for step in plan.steps[next_index:]:
            missing = [dep for dep in (step.depends_on or []) if dep not in available]
            if missing:
                return f"step '{step.step_name}' depends on {missing}, which no earlier step produces"
            available.add(step.step_name)
        return None

    def _accumulate_timing(self, step_result: StepExecutionResult) -> None:
        metrics = step_result.metrics or {}
        self.total_pending_time += metrics.get("pending_time", 0.0)
        self.total_running_time += metrics.get("running_time", 0.0)

    def _build_failure_result(
        self,
        material_id: str,
        step_name: str,
        step_result: StepExecutionResult,
    ) -> Dict[str, Any]:
        return self._failure_payload(material_id, step_result.status, step_name, step_result.error_message)

    def _failure_payload(
        self,
        material_id: str,
        status: str,
        step_name: Optional[str],
        error: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "id": material_id,
            "status": status,
            "failed_step": step_name,
            "error": error,
            "running_time_hours": self.total_running_time / 3600,
            "pending_time_hours": self.total_pending_time / 3600,
            "llm_tokens": self.orchestrator.llm_client.token_budget.used,
        }

    def _save_json(self, path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _append_reasoning_record(self, context: ExecutionContext, record_payload: Optional[Dict[str, Any]]) -> None:
        if not context.reasoning_log or not record_payload:
            return
        payload = dict(record_payload)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload.setdefault("model_name", "")
        payload.setdefault("token_usage", {})
        payload.setdefault("stage", "unknown")
        payload.setdefault("input_context", {})
        payload.setdefault("output_decision", {})
        payload.setdefault("reasoning_trace", "")
        context.reasoning_log.add_record(ReasoningRecord(**payload))

    def _append_monitor_reasoning(self, context: ExecutionContext, step_result: StepExecutionResult) -> None:
        assessment = (step_result.metrics or {}).get("monitor_assessment")
        if not assessment:
            return
        if isinstance(assessment, dict) and assessment.get("reasoning_record"):
            self._append_reasoning_record(context, assessment.get("reasoning_record"))
            return
        self._append_reasoning_record(
            context,
            {
                "stage": "monitoring",
                "input_context": {"step_name": step_result.step_name},
                "output_decision": assessment if isinstance(assessment, dict) else {"assessment": assessment},
                "reasoning_trace": assessment.get("llm_assessment", "") if isinstance(assessment, dict) else "",
            },
        )

    def _post_workflow_extraction(
        self,
        context: ExecutionContext,
        material_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Rule-based property extraction by the Postprocessing Agent."""
        try:
            return self.orchestrator.postprocessing_agent.extract_all_properties(self.work_base_dir, context)
        except Exception as exc:
            logger.warning(f"[{material_id}] Property extraction failed: {exc}")
            return {}

    def _build_success_result(
        self,
        material_id: str,
        task_description: str,
        total_time: float,
        total_steps: int,
        extracted_properties: Dict[str, Any],
        final_result: Optional[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        result = {
            "id": material_id,
            "status": "success",
            "task_description": task_description,
            "execution_time_hours": total_time / 3600,
            "running_time_hours": self.total_running_time / 3600,
            "pending_time_hours": self.total_pending_time / 3600,
            "llm_tokens": self.orchestrator.llm_client.token_budget.used,
            "n_steps_completed": total_steps,
        }
        if extracted_properties:
            result.update(extracted_properties)
        if final_result:
            for key, value in final_result.items():
                result.setdefault(key, value)
        return result
