#!/usr/bin/env bash
# Run VASPBench once for each system configuration with the current MODEL_NAME.
#
# Usage:
#   MODEL_NAME=google/gemini-3.1-pro-preview scripts/run_all_configs.sh
#   scripts/run_all_configs.sh full openloop        # selected configurations only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS=("$@")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  CONFIGS=(rule_based openloop static_plan static_plan_repair full_no_reflector full)
fi

for cfg in "${CONFIGS[@]}"; do
  echo "=== VASPBench | config=${cfg} | model=${MODEL_NAME:-openai/gpt-5.2} ==="
  python "$REPO_ROOT/benchmarks/vaspbench/run_vaspbench.py" --config "$cfg"
done
