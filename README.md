<div align="center">
<h1>AutoDFT: A Closed-Loop Multi-Agent Framework for Autonomous DFT Calculations</h1>
</div>

<div align="center">
<b>Penghui Yang</b> ·
<b>Zhonghan Zhang</b> ·
<b>Yue Li</b> ·
<b>Xinrun Wang</b> ·
<b>Yanchen Deng</b> ·
<b>Yuhao Lu</b> ·
<b>Bijun Tang</b> ·
<b>Zheng Liu</b> ·
<b>Bo An</b>
</div>

<div align="center">
[<a href="https://arxiv.org/abs/2605.26179">Paper</a>]
</div>
<br>

AutoDFT is a closed-loop multi-agent framework for autonomous DFT calculations with VASP.
Given a crystal structure (CIF) and a natural-language task, it plans the workflow, generates
VASP parameters step by step from earlier results, monitors running jobs, recovers from
failures, and revises the plan when intermediate results call for it.
## Architecture

| Agent | Paradigm | Module |
|---|---|---|
| Strategic Planner: skeletal plan of step objectives, no VASP parameters | LLM | `agents/strategic_planner.py` |
| Step Planner: INCAR/KPOINTS for the next step, conditioned on the history | LLM | `agents/step_planner.py` |
| VASP Executor: writes inputs, submits the job | rule-based | `agents/vasp_executor.py` |
| Dual-Path Monitor: rule checks every 10 min, LLM fallback every 5 checks | hybrid | `agents/reasoning_monitor.py`, `agents/rule_monitor.py` |
| Recovery Agent: diagnoses a failed run and proposes fixed parameters | LLM | `agents/reasoning_recovery.py` |
| Step Reflector: accept / redo / modify the plan after each step | LLM | `agents/step_reflector.py` |
| Postprocessing Agent: extracts property values from the outputs | rule-based | `agents/postprocessing_agent.py` |

The control loop (Algorithm 1 in the paper) is in `workflow_controller.py`, and all prompt
templates are in `tools/workflow_prompts.py`.

Default limits, set in `config.py` and overridable through environment variables:

| Setting | Default |
|---|---|
| Per-step VASP wall-clock timeout (`STEP_TIMEOUT_HOURS`) | 4 h |
| Recovery attempts per step (`MAX_RETRIES_PER_STEP`) | 3 |
| Plan revisions per workflow (`MAX_PLAN_REVISIONS`) | 3 |
| VASP steps per workflow (`MAX_TOTAL_STEPS`) | 15 |
| LLM token budget per workflow (`MAX_TOKENS_PER_WORKFLOW`) | 200,000 (exceeding it fails the workflow) |
| Rule-based monitor interval (`MONITOR_CHECK_INTERVAL`) | 600 s |
| LLM monitor fallback (`LLM_MONITOR_INTERVAL`) | every 5 rule checks |

## Requirements

- Python 3.9 or newer, and `pip install -r requirements.txt`.
- **VASP 6.x** under your own license. This repository contains no VASP binaries or POTCAR files.
- **VASPKIT** on `PATH`, with `PBE_PATH` in `~/.vaspkit` pointing to your POTCAR library. AutoDFT
  uses it to convert CIF to POSCAR, generate POTCAR files, and build k-meshes and k-paths.
- An LLM endpoint with an OpenAI-compatible API. The default is OpenRouter.

## Configuration

All machine-specific settings are environment variables:

```bash
export OPENROUTER_API_KEY=...                 # required for every configuration except rule_based
export MODEL_NAME=openai/gpt-5.2              # any OpenRouter model id
export VASP_STD_CMD=/path/to/vasp_std
export VASP_NCL_CMD=/path/to/vasp_ncl         # used automatically for SOC / non-collinear steps
export VASP_GAM_CMD=/path/to/vasp_gam
export VASP_ENV_SCRIPT=/path/to/vasp_env.sh   # optional, sourced before each VASP run
export EXECUTION_MODE=local                   # local | slurm | auto
export GPU_ID=0                               # local mode: CUDA_VISIBLE_DEVICES for VASP
export AUTODFT_WORK_DIR=/path/to/output       # default: ./runs
```

In Slurm mode, set `SLURM_HEADER` to the `#SBATCH` lines your cluster needs, separated by `\n`.

## System configurations

`--config` selects which agents use the LLM. These are the configurations compared in the paper:

| `--config` | Description |
|---|---|
| `rule_based` | Hand-crafted workflow template chosen from the task description; no LLM |
| `openloop` | One LLM call plans the workflow; heuristic parameter templates; rule-based monitoring and recovery |
| `static_plan` | One LLM call generates the workflow **and** all INCAR/KPOINTS upfront; rule-based execution |
| `static_plan_repair` | `static_plan` plus LLM-driven recovery from VASP errors |
| `full_no_reflector` | Full AutoDFT with the Step Reflector disabled |
| `full` | Full AutoDFT (default) |

## Usage

Run a single structure:

```bash
python run_autodft.py --cif my_structure.cif --task "Compute the bandgap" --config full
```

Run VASPBench (34 tasks):

```bash
python benchmarks/vaspbench/run_vaspbench.py --config full
python benchmarks/vaspbench/run_vaspbench.py --config full --task Fcc_Si   # one task
python benchmarks/vaspbench/run_vaspbench.py --list
scripts/run_all_configs.sh                                                  # every configuration
```

Run the Materials Project 20-material subset for one property:

```bash
python benchmarks/materials_project/run_mp_benchmark.py --property bandgap
python benchmarks/materials_project/run_mp_benchmark.py --property magnetic_moment
python benchmarks/materials_project/run_mp_benchmark.py --property formation_energy
```

## Outputs

Each material or task gets its own directory under the output directory:

```
<material_id>/
  skeletal_plan.json      # current (possibly revised) skeletal plan
  reasoning_log.json      # every LLM decision: prompts, outputs, token usage
  results.json            # final status, timing, token usage, extracted properties
  00_<step_name>/         # VASP inputs and outputs for each step
    attempt_<n>_<reason>_<timestamp>/   # archived failed attempts
```

If `results.json` already exists for a material, it is not run again. The benchmark runners also
write a `summary.json` to the run directory.

## Benchmarks

- `benchmarks/vaspbench/`: 34 tasks derived from VASP Wiki examples. Each task has only a
  `structure.cif` and a natural-language objective in `task.txt`. `tasks.json` lists the source page
  and the calculation types. No reference INCAR, KPOINTS, POTCAR or output files are included.
- `benchmarks/materials_project/`: 20 structures from the Materials Project
  (https://materialsproject.org, CC BY 4.0). `materials.json` lists the mp-ids and the task
  description used for each property.

## Notes

- Results vary from run to run because of LLM sampling, and also depend on the backbone model,
  the VASP build, and the POTCAR set.
- Formation energies use the elemental reference energies in `data/elemental_references.json`.
  For quantitative work, recompute them with your own POTCAR set.

## Citation

If you find this repo useful for your research, please consider citing the paper.

```bibtex
@inproceedings{yang2027autodft,
  author={Penghui Yang and Zhonghan Zhang and Yue Li and Xinrun Wang and Yanchen Deng and Yuhao Lu and Bijun Tang and Zheng Liu and Bo An},
  title={AutoDFT: A Closed-Loop Multi-Agent Framework for Autonomous DFT Calculations},
  booktitle={Proceedings of the 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2027},
}
```
