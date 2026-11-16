"""
Configuration for AutoDFT.

Everything that depends on the local machine (VASP binaries, LLM credentials,
output location) is read from environment variables. The workflow limits below
are the values used in the paper (Section 3 and Appendix B.2).
"""
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Output ---
WORK_BASE_DIR = os.path.abspath(os.environ.get("AUTODFT_WORK_DIR", os.path.join(_REPO_ROOT, "runs")))

# --- VASP ---
# Executables for the standard, non-collinear (SOC) and Gamma-only builds.
VASP_STD_CMD = os.environ.get("VASP_STD_CMD", "vasp_std")
VASP_NCL_CMD = os.environ.get("VASP_NCL_CMD", "vasp_ncl")
VASP_GAM_CMD = os.environ.get("VASP_GAM_CMD", "vasp_gam")
# Optional shell script sourced before every VASP run (PATH, LD_LIBRARY_PATH, ...).
VASP_ENV_SCRIPT = os.environ.get("VASP_ENV_SCRIPT", "")
# Optional command run before vaspkit (e.g. "module load vaspkit"); empty if vaspkit is on PATH.
VASPKIT_MODULE_LOAD = os.environ.get("VASPKIT_MODULE_LOAD", "")

# --- Execution ---
# "auto" uses Slurm when available and falls back to local execution.
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "auto")
LOCAL_MPI_PROCESSES = int(os.environ.get("LOCAL_MPI_PROCESSES", "1"))
LOCAL_OMP_THREADS = int(os.environ.get("LOCAL_OMP_THREADS", "8"))
# GPU used for local VASP runs (exported as CUDA_VISIBLE_DEVICES). -1 leaves it unset.
GPU_ID = int(os.environ.get("GPU_ID", "0"))
# Extra #SBATCH lines / module loads for Slurm mode, separated by "\n".
SLURM_HEADER = os.environ.get("SLURM_HEADER", "#SBATCH --ntasks=1\n#SBATCH --gres=gpu:1")

# --- LLM (OpenRouter, OpenAI-compatible API) ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-5.2")

# --- Workflow limits (paper defaults) ---
STEP_TIMEOUT_HOURS = float(os.environ.get("STEP_TIMEOUT_HOURS", "4.0"))  # wall-clock limit per VASP step
MAX_TOKENS_PER_WORKFLOW = int(os.environ.get("MAX_TOKENS_PER_WORKFLOW", "200000"))
MAX_RETRIES_PER_STEP = int(os.environ.get("MAX_RETRIES_PER_STEP", "3"))
MAX_PLAN_REVISIONS = int(os.environ.get("MAX_PLAN_REVISIONS", "3"))  # R_max
MAX_TOTAL_STEPS = int(os.environ.get("MAX_TOTAL_STEPS", "15"))  # N_max
# Dual-path monitor: rule-based check every 10 min, LLM fallback every 5 rule checks.
MONITOR_CHECK_INTERVAL = int(os.environ.get("MONITOR_CHECK_INTERVAL", "600"))
LLM_MONITOR_INTERVAL = int(os.environ.get("LLM_MONITOR_INTERVAL", "5"))

# --- System configurations evaluated in the paper ---
# LLM_LEVEL: "none" (no LLM), "planning_only" (strategic planner only), "full".
# UPFRONT_PARAMS: the strategic planner also emits complete INCAR/KPOINTS for every step.
CONFIGURATIONS = {
    # Hand-crafted workflow template selected from the task description; no LLM.
    "rule_based": dict(LLM_LEVEL="none", CONSERVATIVE_MODE=True, UPFRONT_PARAMS=False,
                       ENABLE_LLM_STEP_PLANNING=False, ENABLE_LLM_MONITORING=False,
                       ENABLE_LLM_RECOVERY=False, ENABLE_LLM_REFLECTION=False),
    # One LLM call plans the workflow; heuristic parameter templates; rule-based execution.
    "openloop": dict(LLM_LEVEL="planning_only", CONSERVATIVE_MODE=False, UPFRONT_PARAMS=False,
                     ENABLE_LLM_STEP_PLANNING=False, ENABLE_LLM_MONITORING=False,
                     ENABLE_LLM_RECOVERY=False, ENABLE_LLM_REFLECTION=False),
    # One LLM call generates the workflow and all INCAR/KPOINTS upfront.
    "static_plan": dict(LLM_LEVEL="planning_only", CONSERVATIVE_MODE=False, UPFRONT_PARAMS=True,
                        ENABLE_LLM_STEP_PLANNING=False, ENABLE_LLM_MONITORING=False,
                        ENABLE_LLM_RECOVERY=False, ENABLE_LLM_REFLECTION=False),
    # Static-Plan plus LLM-driven recovery from VASP errors.
    "static_plan_repair": dict(LLM_LEVEL="full", CONSERVATIVE_MODE=False, UPFRONT_PARAMS=True,
                               ENABLE_LLM_STEP_PLANNING=False, ENABLE_LLM_MONITORING=False,
                               ENABLE_LLM_RECOVERY=True, ENABLE_LLM_REFLECTION=False),
    # Full AutoDFT with the Step Reflector disabled.
    "full_no_reflector": dict(LLM_LEVEL="full", CONSERVATIVE_MODE=False, UPFRONT_PARAMS=False,
                              ENABLE_LLM_STEP_PLANNING=True, ENABLE_LLM_MONITORING=True,
                              ENABLE_LLM_RECOVERY=True, ENABLE_LLM_REFLECTION=False),
    # Full AutoDFT.
    "full": dict(LLM_LEVEL="full", CONSERVATIVE_MODE=False, UPFRONT_PARAMS=False,
                 ENABLE_LLM_STEP_PLANNING=True, ENABLE_LLM_MONITORING=True,
                 ENABLE_LLM_RECOVERY=True, ENABLE_LLM_REFLECTION=True),
}
DEFAULT_CONFIGURATION = os.environ.get("AUTODFT_CONFIG", "full")

_active = CONFIGURATIONS[DEFAULT_CONFIGURATION]
LLM_LEVEL = _active["LLM_LEVEL"]
CONSERVATIVE_MODE = _active["CONSERVATIVE_MODE"]
UPFRONT_PARAMS = _active["UPFRONT_PARAMS"]
ENABLE_LLM_STEP_PLANNING = _active["ENABLE_LLM_STEP_PLANNING"]
ENABLE_LLM_MONITORING = _active["ENABLE_LLM_MONITORING"]
ENABLE_LLM_RECOVERY = _active["ENABLE_LLM_RECOVERY"]
ENABLE_LLM_REFLECTION = _active["ENABLE_LLM_REFLECTION"]


def get_configuration(name: str) -> dict:
    """Return the agent settings for one of the named system configurations."""
    if name not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration '{name}'. Choose from: {', '.join(CONFIGURATIONS)}")
    return dict(CONFIGURATIONS[name], CONFIGURATION_NAME=name)
