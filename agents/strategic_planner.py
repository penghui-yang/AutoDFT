"""
Strategic skeletal planner for the adaptive workflow.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from typing import Any, Dict, Optional

from agents.hardcoded_workflows import get_hardcoded_workflow
from agents.llm_client import LLMClient, TokenBudgetExceeded
from common_types.workflow_models import SkeletalPlan, SkeletalStep
from config import LLM_LEVEL, UPFRONT_PARAMS
from execution.job_manager import run_command
from tools.file_parsers import get_formula_from_poscar
from tools.workflow_prompts import build_strategic_planner_prompt

logger = logging.getLogger(__name__)


class StrategicPlanner:
    """Generate a skeletal plan and preserve fallback compatibility."""

    def __init__(self, work_base_dir: str, config: Optional[Dict[str, Any]] = None, llm_client: Optional[LLMClient] = None):
        self.work_base_dir = work_base_dir
        self.config = config or {}
        self.llm_client = llm_client or LLMClient(self.config)

    def plan(
        self,
        task_description: str,
        material_id: str,
        cif_path: str,
        material_type: str = "auto",
        additional_context: str = "",
        conservative_mode: bool = False,
    ) -> SkeletalPlan:
        material_type, formula, poscar_content = self._prepare_material_context(
            material_id, cif_path, material_type
        )

        if conservative_mode or self.config.get("LLM_LEVEL", LLM_LEVEL) == "none":
            hardcoded = get_hardcoded_workflow(task_description, material_id, material_type)
            if hardcoded:
                plan = hardcoded.to_skeletal_plan()
                plan.material_formula = formula
                plan.metadata.setdefault("source", "hardcoded_workflow")
                plan.metadata.setdefault(
                    "reasoning_record",
                    {
                        "stage": "strategic_planning",
                        "input_context": {"source": "hardcoded_workflow"},
                        "output_decision": {"source": "hardcoded_workflow"},
                        "reasoning_trace": "Used hardcoded workflow fallback.",
                    },
                )
                return plan

        if self.config.get("LLM_LEVEL", LLM_LEVEL) != "none":
            try:
                system_prompt, user_prompt = build_strategic_planner_prompt(
                    task_description=task_description,
                    material_id=material_id,
                    formula=formula,
                    poscar_content=poscar_content,
                    material_type=material_type,
                    additional_context=additional_context,
                    upfront_params=self.config.get("UPFRONT_PARAMS", UPFRONT_PARAMS),
                )
                response_text, record = self.llm_client.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    stage="strategic_planning",
                    temperature=0.3,
                )
                payload = json.loads(response_text)
                steps = [SkeletalStep(**step) for step in payload.get("steps", [])]
                return SkeletalPlan(
                    task_description=payload.get("task_description", task_description),
                    material_formula=payload.get("material_formula", formula),
                    material_type=payload.get("material_type", material_type),
                    overall_strategy=payload.get("overall_strategy", ""),
                    steps=steps,
                    global_considerations=payload.get("global_considerations", []),
                    reasoning_trace=payload.get("reasoning_trace", ""),
                    material_id=material_id,
                    metadata={
                        "reasoning_record": asdict(record),
                        "source": "llm_strategic_planner",
                        "upfront_params": bool(self.config.get("UPFRONT_PARAMS", UPFRONT_PARAMS)),
                    },
                )
            except TokenBudgetExceeded:
                raise
            except Exception as exc:
                logger.warning(f"Strategic planner LLM path failed, falling back to heuristic skeletal plan: {exc}")

        return self._heuristic_plan(
            task_description=task_description,
            material_id=material_id,
            formula=formula,
            material_type=material_type,
        )

    def _prepare_material_context(self, material_id: str, cif_path: str, material_type: str) -> tuple[str, str, str]:
        temp_dir = os.path.join(self.work_base_dir, f"{material_id}_strategic_temp")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            cif_file = os.path.basename(cif_path)
            shutil.copy(cif_path, temp_dir)
            run_command(f'printf "{cif_file}\\n\\n" | vaspkit -task 105', cwd=temp_dir, shell=True)
            poscar_path = os.path.join(temp_dir, "POSCAR")
            with open(poscar_path, "r") as handle:
                poscar_content = handle.read()
            formula = get_formula_from_poscar(poscar_path)
            if material_type == "auto":
                material_type = self._infer_material_type(poscar_content)
            return material_type, formula, poscar_content
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _infer_material_type(self, poscar_content: str) -> str:
        lines = poscar_content.strip().splitlines()
        if len(lines) < 5:
            return "3D bulk"
        try:
            lattice_vectors = []
            for i in range(2, 5):
                lattice_vectors.append([float(x) for x in lines[i].split()])
            c_length = sum(x ** 2 for x in lattice_vectors[2]) ** 0.5
            if c_length > 15 and max(abs(lattice_vectors[0][2]), abs(lattice_vectors[1][2])) < 1.0:
                return "2D"
        except (ValueError, IndexError):
            pass
        return "3D bulk"

    def _heuristic_plan(
        self,
        task_description: str,
        material_id: str,
        formula: str,
        material_type: str,
    ) -> SkeletalPlan:
        task_lower = task_description.lower()
        steps = [
            SkeletalStep(
                step_name="relaxation",
                objective="Relax the structure before property evaluation.",
                computation_type="relaxation",
                success_criteria={"converged": True},
                priority="required",
            )
        ]
        if "band" in task_lower or "gap" in task_lower:
            steps.append(
                SkeletalStep(
                    step_name="band_structure",
                    objective="Compute band structure or bandgap-relevant electronic states.",
                    computation_type="band",
                    depends_on=["relaxation"],
                    success_criteria={"eigenvalues_available": True},
                    priority="required",
                )
            )
        elif "dos" in task_lower:
            steps.append(
                SkeletalStep(
                    step_name="static_scf",
                    objective="Compute a converged static charge density for DOS analysis.",
                    computation_type="static",
                    depends_on=["relaxation"],
                    success_criteria={"charge_density_available": True},
                    priority="required",
                )
            )
            steps.append(
                SkeletalStep(
                    step_name="dos",
                    objective="Extract the density of states from the static calculation.",
                    computation_type="dos",
                    depends_on=["static_scf"],
                    success_criteria={"dos_available": True},
                    priority="required",
                )
            )
        return SkeletalPlan(
            task_description=task_description,
            material_formula=formula,
            material_type=material_type,
            overall_strategy="Heuristic fallback skeletal plan.",
            steps=steps,
            global_considerations=[],
            reasoning_trace="Used heuristic strategic fallback.",
            material_id=material_id,
            metadata={
                "source": "heuristic_strategic_plan",
                "reasoning_record": {
                    "stage": "strategic_planning",
                    "input_context": {"task_description": task_description},
                    "output_decision": {"steps": [step.step_name for step in steps]},
                    "reasoning_trace": "Used heuristic strategic fallback.",
                },
            },
        )
