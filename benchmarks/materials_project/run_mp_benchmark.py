#!/usr/bin/env python3
"""
Run AutoDFT on the 20-material Materials Project subset for one property.

Properties and the task descriptions given to AutoDFT:
    bandgap           "Compute the bandgap"
    magnetic_moment   "Calculate the total magnetic moment"
    formation_energy  "Compute the heat of formation"

Usage:
    python benchmarks/materials_project/run_mp_benchmark.py --property bandgap
    python benchmarks/materials_project/run_mp_benchmark.py --property magnetic_moment --material mp-19009
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import CONFIGURATIONS, DEFAULT_CONFIGURATION, MODEL_NAME, STEP_TIMEOUT_HOURS, WORK_BASE_DIR, get_configuration
from logging_setup import setup_logging
from workflow_controller import WorkflowController

logger = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parent


def main() -> None:
    with open(BENCH_DIR / "materials.json") as handle:
        benchmark = json.load(handle)
    properties = benchmark["properties"]

    parser = argparse.ArgumentParser(description="Run AutoDFT on the Materials Project 20-material subset")
    parser.add_argument("--property", required=True, choices=list(properties))
    parser.add_argument("--config", default=DEFAULT_CONFIGURATION, choices=list(CONFIGURATIONS))
    parser.add_argument("--material", default=None, help="Run a single material (mp-id)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: AUTODFT_WORK_DIR/materials_project/<property>_<config>_<model>_<timestamp>)")
    parser.add_argument("--timeout", type=float, default=STEP_TIMEOUT_HOURS, help="Per-step VASP timeout in hours")
    args = parser.parse_args()

    materials = benchmark["materials"]
    if args.material:
        materials = [m for m in materials if m["material_id"] == args.material]
        if not materials:
            sys.exit(f"Unknown material: {args.material}")

    model_slug = MODEL_NAME.replace("/", "_").replace(":", "_")
    output_dir = Path(args.output_dir) if args.output_dir else Path(WORK_BASE_DIR) / "materials_project" / (
        f"{args.property}_{args.config}_{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir / "autodft.log"))
    logger.info(f"MP benchmark | property={args.property} | model={MODEL_NAME} | config={args.config}")

    run_config = get_configuration(args.config)
    controller = WorkflowController(str(output_dir), config=run_config)

    results = []
    for i, material in enumerate(materials, 1):
        logger.info(f"[{i}/{len(materials)}] {material['material_id']} ({material['formula']})")
        result = controller.run_workflow(
            material_id=material["material_id"],
            task_description=properties[args.property],
            cif_path=str(BENCH_DIR / material["structure"]),
            step_timeout_hours=args.timeout,
            conservative_mode=run_config["CONSERVATIVE_MODE"],
        )
        results.append({"material_id": material["material_id"], "formula": material["formula"],
                         **(result or {"status": "failed"})})

    n_success = sum(1 for r in results if r.get("status") == "success")
    summary = {
        "benchmark": "Materials Project 20-material subset",
        "property": args.property,
        "model": MODEL_NAME,
        "configuration": run_config,
        "n_materials": len(results),
        "n_success": n_success,
        "results": results,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"{args.property}: {n_success}/{len(results)} workflows succeeded. Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
