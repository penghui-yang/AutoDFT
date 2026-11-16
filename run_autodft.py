#!/usr/bin/env python3
"""
Run AutoDFT on a single structure.

Example:
    python run_autodft.py --cif Si.cif --task "Compute the bandgap of bulk silicon"
    python run_autodft.py --cif Si.cif --task-file task.txt --config openloop --id Si
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIGURATIONS, DEFAULT_CONFIGURATION, MODEL_NAME, STEP_TIMEOUT_HOURS, WORK_BASE_DIR, get_configuration
from logging_setup import setup_logging
from tools.supported_properties import check_task_support
from workflow_controller import WorkflowController

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AutoDFT workflow for one structure.")
    parser.add_argument("--cif", required=True, help="Input crystal structure (CIF)")
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task", help="Natural-language task description")
    task.add_argument("--task-file", help="File containing the task description")
    parser.add_argument("--id", default=None, help="Material identifier (default: CIF file name)")
    parser.add_argument("--config", default=DEFAULT_CONFIGURATION, choices=list(CONFIGURATIONS),
                        help=f"System configuration (default: {DEFAULT_CONFIGURATION})")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: AUTODFT_WORK_DIR/<config>)")
    parser.add_argument("--timeout", type=float, default=STEP_TIMEOUT_HOURS, help="Per-step VASP timeout in hours")
    args = parser.parse_args()

    task_description = args.task if args.task else Path(args.task_file).read_text().strip()
    cif_path = os.path.abspath(args.cif)
    material_id = args.id or Path(cif_path).stem
    output_dir = os.path.abspath(args.output_dir or os.path.join(WORK_BASE_DIR, args.config))
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, "autodft.log"))

    supported, matched, _ = check_task_support(task_description)
    if not supported:
        logger.warning("Task does not match any supported property keyword; running anyway.")
    logger.info(f"Model: {MODEL_NAME} | configuration: {args.config} | material: {material_id}")

    run_config = get_configuration(args.config)
    controller = WorkflowController(output_dir, config=run_config)
    result = controller.run_workflow(
        material_id=material_id,
        task_description=task_description,
        cif_path=cif_path,
        step_timeout_hours=args.timeout,
        conservative_mode=run_config["CONSERVATIVE_MODE"],
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
