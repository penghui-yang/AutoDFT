#!/usr/bin/env python3
"""
Run AutoDFT on VASPBench (34 tasks derived from the official VASP documentation).

Each task supplies only a structure (structure.cif) and a natural-language
objective (task.txt). A task counts as successful when the workflow completes
all of its steps with converged, validated VASP outputs (status "success").

Usage:
    python benchmarks/vaspbench/run_vaspbench.py --config full
    python benchmarks/vaspbench/run_vaspbench.py --config openloop --task Fcc_Si
    python benchmarks/vaspbench/run_vaspbench.py --list
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import CONFIGURATIONS, DEFAULT_CONFIGURATION, MODEL_NAME, STEP_TIMEOUT_HOURS, WORK_BASE_DIR, get_configuration
from logging_setup import setup_logging
from workflow_controller import WorkflowController
from common_types.workflow_models import ExecutionContext
from benchmarks.vaspbench.extract_properties import extract_all_properties

logger = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parent
TASKS_DIR = BENCH_DIR / "tasks"
TASKS_FILE = BENCH_DIR / "tasks.json"


class VASPBenchController(WorkflowController):
    """Extracts properties from the last completed calculation of each task."""

    def _post_workflow_extraction(self, context: ExecutionContext, material_id: str, **kwargs) -> Dict[str, Any]:
        for step_name in reversed(list(context.completed_steps)):
            calc_dir = context.completed_steps[step_name]
            if os.path.exists(os.path.join(calc_dir, "OUTCAR")):
                try:
                    return {"extracted_properties": extract_all_properties(calc_dir, material_id)}
                except Exception as exc:
                    logger.warning(f"Property extraction failed for {material_id}: {exc}")
                    break
        return {}


def load_tasks() -> list:
    with open(TASKS_FILE) as handle:
        return json.load(handle)["tasks"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoDFT on VASPBench")
    parser.add_argument("--config", default=DEFAULT_CONFIGURATION, choices=list(CONFIGURATIONS))
    parser.add_argument("--task", default=None, help="Run a single task (directory name)")
    parser.add_argument("--list", action="store_true", help="List tasks and exit")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: AUTODFT_WORK_DIR/vaspbench/<config>_<model>_<timestamp>)")
    parser.add_argument("--timeout", type=float, default=STEP_TIMEOUT_HOURS, help="Per-step VASP timeout in hours")
    args = parser.parse_args()

    tasks = load_tasks()
    if args.list:
        for task in tasks:
            print(f"{task['id']:>2}  {task['dir']:<75} {', '.join(task['calculation_types'])}")
        return
    if args.task:
        tasks = [t for t in tasks if t["dir"] == args.task]
        if not tasks:
            sys.exit(f"Unknown task: {args.task}")

    model_slug = MODEL_NAME.replace("/", "_").replace(":", "_")
    output_dir = Path(args.output_dir) if args.output_dir else Path(WORK_BASE_DIR) / "vaspbench" / (
        f"{args.config}_{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir / "autodft.log"))
    logger.info(f"VASPBench | model={MODEL_NAME} | config={args.config} | output={output_dir}")

    run_config = get_configuration(args.config)
    controller = VASPBenchController(str(output_dir), config=run_config)

    results = []
    for i, task in enumerate(tasks, 1):
        task_dir = TASKS_DIR / task["dir"]
        logger.info(f"[{i}/{len(tasks)}] {task['dir']}")
        result: Optional[Dict[str, Any]] = controller.run_workflow(
            material_id=task["dir"],
            task_description=(task_dir / "task.txt").read_text().strip(),
            cif_path=str(task_dir / "structure.cif"),
            step_timeout_hours=args.timeout,
            conservative_mode=run_config["CONSERVATIVE_MODE"],
        )
        results.append({"task": task["dir"], **(result or {"status": "failed"})})

    n_success = sum(1 for r in results if r.get("status") == "success")
    summary = {
        "benchmark": "VASPBench",
        "model": MODEL_NAME,
        "configuration": run_config,
        "step_timeout_hours": args.timeout,
        "n_tasks": len(results),
        "n_success": n_success,
        "results": results,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"VASPBench: {n_success}/{len(results)} tasks succeeded. Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
