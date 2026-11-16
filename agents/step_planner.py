"""
Runtime step planner that materializes concrete VASP parameters per step.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from agents.llm_client import LLMClient, TokenBudgetExceeded
from common_types.workflow_models import ExecutionContext, SkeletalStep, StepParameters, WorkflowStep
from config import ENABLE_LLM_STEP_PLANNING, LLM_LEVEL
from tools.workflow_prompts import build_step_planner_prompt

logger = logging.getLogger(__name__)


def _parse_incar_file(incar_path: str) -> Dict[str, str]:
    """Parse a VASP INCAR into a {KEY: raw_value_string} dict. Comments and blanks skipped."""
    result: Dict[str, str] = {}
    try:
        with open(incar_path, "r") as f:
            for raw in f:
                line = raw.split("#", 1)[0].split("!", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                result[key.strip().upper()] = val.strip()
    except OSError:
        pass
    return result


def _summarize_kpoints_file(kpoints_path: str) -> str:
    """Return a one-to-two-line summary of a KPOINTS file."""
    try:
        with open(kpoints_path, "r") as f:
            lines = [ln.rstrip("\n") for ln in f.readlines()]
    except OSError:
        return "(not written)"
    if not lines:
        return "(empty)"
    header = lines[0].strip()
    scheme = lines[2].strip() if len(lines) > 2 else ""
    grid_or_count = lines[3].strip() if len(lines) > 3 else ""
    if scheme.lower().startswith("line"):
        n_pts = grid_or_count or "?"
        return f"Line-Mode, {n_pts} intersections per segment (header: {header})"
    return f"{scheme or 'auto'} {grid_or_count}".strip() + f"  (header: {header})"


def _collect_prior_step_inputs(context: ExecutionContext) -> List[Dict[str, Any]]:
    """Walk completed VASP steps in execution order and read their actual INCAR/KPOINTS."""
    prior: List[Dict[str, Any]] = []
    for step_name, step_dir in context.vasp_completed_steps.items():
        if not step_dir or not os.path.isdir(step_dir):
            continue
        incar = _parse_incar_file(os.path.join(step_dir, "INCAR"))
        kpts = _summarize_kpoints_file(os.path.join(step_dir, "KPOINTS"))
        prior.append({
            "step_name": step_name,
            "incar": incar,
            "kpoints_summary": kpts,
        })
    return prior


class StepPlanner:
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client: Optional[LLMClient] = None):
        self.config = config or {}
        self.llm_client = llm_client or LLMClient(self.config)

    def evaluate_condition(self, skeletal_step: SkeletalStep, step_outcomes: Dict[str, Any]) -> bool:
        """Minimal condition evaluation; defaults to execute when unsure."""
        if not skeletal_step.condition:
            return True
        condition = skeletal_step.condition.lower()
        if "magnetic" in condition:
            for outcome in step_outcomes.values():
                if outcome.metadata.get("is_magnetic") or outcome.metadata.get("total_magnetic_moment", 0):
                    return True
            return False
        return True

    def plan_step(
        self,
        skeletal_step: SkeletalStep,
        context: ExecutionContext,
    ) -> StepParameters:
        if skeletal_step.predefined_parameters is not None:
            params = skeletal_step.predefined_parameters
            if isinstance(params, dict):
                # predefined_parameters can arrive as a plain dict when the step_reflector
                # reconstructs a modified plan from LLM JSON (SkeletalStep(**step_dict)).
                # Convert it back to StepParameters, ignoring any extra LLM-added keys.
                valid_fields = StepParameters.__dataclass_fields__
                try:
                    filtered = {k: v for k, v in params.items() if k in valid_fields}
                    params = StepParameters(**filtered)
                except Exception:
                    return self._heuristic_parameters(skeletal_step)
                # If the LLM returned a free-form dict (e.g. {"recommended": {...}})
                # none of its keys survive the filter, leaving incar={}.
                # An empty INCAR means VASP will never be submitted — fall back to heuristic.
                if not params.incar:
                    logger.warning(
                        f"predefined_parameters for '{skeletal_step.step_name}' yielded empty INCAR "
                        f"(raw keys: {list(skeletal_step.predefined_parameters.keys())}), "
                        f"falling back to heuristic parameters."
                    )
                    return self._heuristic_parameters(skeletal_step)
            return params

        if (self.config.get("LLM_LEVEL", LLM_LEVEL) in {"none", "planning_only"}
                or not self.config.get("ENABLE_LLM_STEP_PLANNING", ENABLE_LLM_STEP_PLANNING)):
            return self._heuristic_parameters(skeletal_step)

        try:
            system_prompt, user_prompt = build_step_planner_prompt(
                skeletal_step=asdict(skeletal_step),
                material_context={
                    "material_id": context.material_id,
                    "formula": context.formula,
                    "metadata": context.metadata,
                    "global_settings": asdict(context.global_settings) if context.global_settings else {},
                },
                prior_outcomes=[asdict(outcome) for outcome in context.step_outcomes.values()],
                prior_step_inputs=_collect_prior_step_inputs(context),
            )
            response_text, record = self.llm_client.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage="step_planning",
                temperature=0.2,
            )
            payload = json.loads(response_text)
            params = StepParameters(**payload)
            params.metadata.setdefault("reasoning_record", asdict(record))
            return params
        except TokenBudgetExceeded:
            raise
        except Exception as exc:
            logger.warning(f"Step planner LLM failed for {skeletal_step.step_name}, using heuristic fallback: {exc}")
            return self._heuristic_parameters(skeletal_step)

    def materialize_workflow_step(
        self,
        skeletal_step: SkeletalStep,
        step_parameters: StepParameters,
    ) -> WorkflowStep:
        metadata = step_parameters.metadata or {}
        return WorkflowStep(
            name=skeletal_step.step_name,
            description=skeletal_step.objective,
            input_files_needed=metadata.get("input_files_needed", []),
            output_files_produced=metadata.get("output_files_produced", []),
            INCAR=dict(step_parameters.incar or {}),
            KPOINTS=step_parameters.kpoints,
            POTCAR_notes=metadata.get("POTCAR_notes", ""),
            vaspkit_actions=metadata.get("vaspkit_actions", []),
            sanity_checks=metadata.get("sanity_checks", []),
            expected_time_estimate=metadata.get("expected_time_estimate", ""),
            depends_on=list(skeletal_step.depends_on),
            agent_type=metadata.get("agent_type", "computing"),
            additional_params={
                "constraints": skeletal_step.constraints,
                "success_criteria": skeletal_step.success_criteria,
                "vasp_version": step_parameters.vasp_version,
                **metadata.get("additional_params", {}),
            },
            can_parallelize=metadata.get("can_parallelize", False),
            condition=skeletal_step.condition,
            branch_key=metadata.get("branch_key"),
            is_parallel_set=metadata.get("is_parallel_set", False),
            parallel_count=metadata.get("parallel_count", 1),
            file_dependencies=metadata.get("file_dependencies", {}),
        )

    def _heuristic_parameters(self, skeletal_step: SkeletalStep) -> StepParameters:
        name = skeletal_step.step_name.lower()
        incar: Dict[str, Any] = {"EDIFF": 1e-5, "ENCUT": 520}
        kpoints: Any = "Gamma-centered 6x6x6"
        if "relax" in name:
            incar.update({"IBRION": 2, "NSW": 200, "ISIF": 3, "EDIFFG": -0.02})
        elif "band" in name:
            incar.update({"NSW": 0, "ICHARG": 11})
            kpoints = "high-symmetry k-path"
        elif "static" in name or "scf" in name:
            incar.update({"NSW": 0, "IBRION": -1})
        return StepParameters(
            incar=incar,
            kpoints=kpoints,
            additional_files={},
            vasp_version="std",
            reasoning_trace="Heuristic fallback parameters.",
            confidence=0.4,
            fallback_parameters=None,
            metadata={
                "agent_type": "computing",
                "reasoning_record": {
                    "stage": "step_planning",
                    "input_context": {"step_name": skeletal_step.step_name},
                    "output_decision": {"heuristic": True, "incar": incar, "kpoints": kpoints},
                    "reasoning_trace": "Used heuristic fallback parameters.",
                },
            },
        )
