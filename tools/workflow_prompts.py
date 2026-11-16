"""
LLM prompt templates for the AutoDFT agents (reproduced in Appendix B.5 of the paper):
strategic planner, step planner, reasoning monitor, reasoning recovery and step reflector.
"""

from __future__ import annotations

import json


VASP_KNOWLEDGE_BASE = """
VASP parameter guidance:
- Use `ISPIN=2` and explicit `MAGMOM` for materials likely containing open-shell transition metals.
- For 2D materials prefer `ISIF=2`; for bulk relaxations prefer `ISIF=3`.
- Metallic systems generally use `ISMEAR=1 or 2`; semiconductors use `ISMEAR=0` and small `SIGMA`.
- Difficult convergence can require `ALGO=Normal` or `ALGO=All`, smaller `POTIM`, and conservative mixing.
- SOC or non-collinear calculations require `vasp_ncl`.
- Dense k-mesh and tighter `EDIFF` belong in static/band/DOS steps, not cheap pre-relaxation.
"""

STRATEGIC_PLANNER_SYSTEM_PROMPT = """You are a DFT workflow strategist.
Output JSON only. Produce a skeletal plan: decide step sequence, objectives,
constraints, success criteria, and scientific rationale. Do not choose detailed
INCAR or KPOINTS values unless they are unavoidable hard constraints.

CRITICAL PLANNING CONSTRAINTS:
1. EFFICIENCY: Each step must advance the calculation toward the goal. Do NOT
   include pure validation, verification, integrity-check, or checkpoint steps
   that run a VASP calculation solely to confirm outputs are parseable. Sanity
   checks belong as success_criteria within computation steps, not as separate
   steps. Target 3-6 steps for standard tasks (relaxation, bandgap, DOS, elastic
   constants). Only exceed this for genuinely multi-stage methods (GW, NEB, DFPT).
2. SINGLE-STRUCTURE EXECUTION: The execution engine runs exactly ONE VASP
   calculation per step. If a workflow requires calculations at multiple
   structures (e.g., EOS fitting with N volume points), each volume point must
   be its own step. When only the equilibrium volume is needed, prefer ISIF=3
   variable-cell relaxation over manual EOS fitting — it is cheaper and sufficient
   for most purposes.
3. NO REDUNDANT SINGLE-POINTS: Do not add a preliminary single-point calculation
   before a relaxation just to "establish a baseline" — the first ionic step of
   the relaxation already provides this information.

COMPUTE BUDGET AWARENESS:
- Each VASP step has a hard wall-time limit of 4 hours. If a step cannot
  realistically finish one ionic iteration (or one SCF) within that budget, the
  whole workflow fails and no result is reported. Plan accordingly.
- Relative costs (rough rule of thumb, ceteris paribus):
    PBE / PBEsol / LDA           1×   (baseline)
    DFT+U                       ~1×
    SOC (LSORBIT=.TRUE.)        ~3-4× (spinor basis; also roughly doubles memory)
    meta-GGA (SCAN, TPSS, MBJ)  ~2-3×
    Hybrid (HSE, PBE0, B3LYP)   ~30-100×  (depends on k-mesh and NKRED)
    Hybrid + SOC                ~100-300× (often infeasible on non-trivial cells)
    GW                          ~1000× and up
- Concrete implications:
    * For bandgap tasks: the default is PBE (+ SOC only when heavy elements
      demand it, e.g. Bi/Pb/Hg/Tl/I/Te in systems where SOC qualitatively
      changes the gap). A three-step workflow (relax → SCF → NSCF band path)
      is usually the right answer.
    * Do NOT tack on "optional" HSE / hybrid / mBJ refinement steps "for
      higher accuracy" unless the user task explicitly demands it. These steps
      almost always exceed the 4-hour budget on the kind of bulk cells seen
      here, and if they time out the entire material is logged as a failure.
    * If the user task truly requires hybrid-level accuracy, say so in
      `reasoning_trace` and keep the workflow short (relax-PBE → HSE-SCF on a
      coarse k-mesh), rather than stacking PBE + SOC + HSE + HSE-SOC on top of
      each other.
4. CONDITIONAL STEPS ARE USUALLY EXECUTED: the orchestrator does not have a
   robust "skip if not needed" check for free-form `condition` strings, so any
   step you include in the plan should be expected to actually run. Prefer
   leaving optional refinement steps out of the plan entirely and letting the
   reflector add them on evidence of need, rather than listing them up front
   as `condition: "run only if ..."`."""

# Upfront variant (Static-Plan baselines): the planner must
# commit to complete VASP parameters for every step in the same single call,
# because no step planner runs later.
STRATEGIC_PLANNER_UPFRONT_SYSTEM_PROMPT = STRATEGIC_PLANNER_SYSTEM_PROMPT.replace(
    """Produce a skeletal plan: decide step sequence, objectives,
constraints, success criteria, and scientific rationale. Do not choose detailed
INCAR or KPOINTS values unless they are unavoidable hard constraints.""",
    """Produce a complete workflow plan: decide step sequence,
objectives, constraints, success criteria, and scientific rationale, AND specify the
complete VASP parameters (INCAR and KPOINTS) for EVERY step now. No further planning
happens after this call: each step will be executed exactly with the parameters you
give here, so they must be complete, internally consistent across steps, and
appropriate for this specific material and task.""",
).replace(
    """Prefer
   leaving optional refinement steps out of the plan entirely and letting the
   reflector add them on evidence of need, rather than listing them up front
   as `condition: "run only if ..."`.""",
    """Prefer
   leaving optional refinement steps out of the plan entirely rather than
   listing them up front as `condition: "run only if ..."`.""",
)

STEP_PLANNER_SYSTEM_PROMPT = """You are a VASP step planner.
Output JSON only with concrete VASP parameters for one step.
Ground decisions in prior step outcomes and the step objective.
If uncertainty is high, include fallback parameters and say why."""

REASONING_MONITOR_SYSTEM_PROMPT = """You are monitoring a running VASP calculation.
Output JSON only. Decide one of: continue, converged, intervene, terminate.
Prefer conservative actions and cite the observed convergence behavior."""

REASONING_RECOVERY_SYSTEM_PROMPT = """You are recovering a failed VASP calculation.
Output JSON only. Diagnose the likely cause, propose parameter/file changes,
and keep the fix minimal but technically justified.

When POSITIVE_ENERGY is reported, reason step by step:
1. What is the electronic structure of this material? Metal, semiconductor, or insulator?
2. Check ISMEAR in the current INCAR. ISMEAR=1 or 2 (Methfessel-Paxton / Fermi) is designed
   for metals. Using it on a semiconductor or insulator can produce unphysically large entropy
   contributions and positive total energies. If the material is NOT a metal, ISMEAR should be
   0 (Gaussian smearing) with SIGMA <= 0.05 eV.
3. Check for atomic overlaps: are any interatomic distances < 1.5 Å? If so, the geometry
   itself is unphysical and must be fixed (scale or perturb POSCAR) before changing INCAR.
4. Only after ruling out geometry issues and smearing mismatch should you adjust convergence
   parameters (ALGO, AMIX, NELM, etc.)."""

STEP_REFLECTOR_SYSTEM_PROMPT = """You are reflecting on a completed DFT step.
Output JSON only. Decide one of: accept, redo, modify_plan.
Judge physical plausibility and whether the downstream plan should change."""

STEP_OUTCOME_TEMPLATE = """
## Step: {step_name}
- Converged: {converged}
- Total energy: {total_energy}
- Max force: {max_force}
- Electronic steps per ionic step (last 5): {e_steps}
- Cell volume change: {volume_change}
- Bandgap estimate: {bandgap}
- Warnings: {warnings}
"""


def format_step_outcomes_for_prompt(step_outcomes: list[dict]) -> str:
    """Render prior step outcomes into a compact prompt block."""
    blocks = []
    for outcome in step_outcomes:
        blocks.append(
            STEP_OUTCOME_TEMPLATE.format(
                step_name=outcome.get("step_name"),
                converged=outcome.get("converged"),
                total_energy=outcome.get("total_energy"),
                max_force=outcome.get("max_force"),
                e_steps=outcome.get("electronic_steps_per_ionic", [])[-5:],
                volume_change=outcome.get("cell_volume_change_pct"),
                bandgap=outcome.get("bandgap_estimate"),
                warnings=", ".join(outcome.get("warnings", [])) or "None",
            ).strip()
        )
    return "\n\n".join(blocks) if blocks else "No previous step outcomes."


def build_strategic_planner_prompt(
    task_description: str,
    material_id: str,
    formula: str,
    poscar_content: str,
    material_type: str,
    additional_context: str = "",
    upfront_params: bool = False,
) -> tuple[str, str]:
    user_prompt = f"""Task: {task_description}
Material ID: {material_id}
Formula: {formula}
Material type: {material_type}
Additional context: {additional_context or 'None'}

POSCAR:
{poscar_content}

IMPORTANT: Aim for at most 6 steps. Every step must run a meaningful VASP
calculation that advances toward the goal. Do not add separate verification,
integrity-check, or baseline-confirmation steps.

Return JSON with keys:
- task_description
- material_formula
- material_type
- overall_strategy
- global_considerations
- reasoning_trace
- steps: list of objects with step_name, objective, computation_type, depends_on, constraints, success_criteria, priority, condition
"""
    if not upfront_params:
        return STRATEGIC_PLANNER_SYSTEM_PROMPT, user_prompt

    user_prompt = user_prompt.rstrip("\n") + f""", predefined_parameters

Each step's predefined_parameters must be an object with keys:
- incar: complete INCAR as a JSON object (tag -> value)
- kpoints: KPOINTS specification for this step
- additional_files: object (may be empty)
- vasp_version: "std", "gam" or "ncl"
- reasoning_trace
- confidence

Knowledge base:
{VASP_KNOWLEDGE_BASE}
{CROSS_STEP_CONSISTENCY_NOTES}
"""
    return STRATEGIC_PLANNER_UPFRONT_SYSTEM_PROMPT, user_prompt


assert STRATEGIC_PLANNER_UPFRONT_SYSTEM_PROMPT.count("Do not choose detailed") == 0
assert "reflector" not in STRATEGIC_PLANNER_UPFRONT_SYSTEM_PROMPT


def format_prior_step_inputs_for_prompt(prior_step_inputs: list[dict]) -> str:
    """Render prior steps' actual INCAR/KPOINTS so the planner can keep
    grid-sensitive settings (ENCUT, PREC, ISPIN, LSORBIT, KPOINTS, ...)
    consistent across steps. Only the keys that matter for cross-step
    compatibility are shown to keep the prompt compact."""
    if not prior_step_inputs:
        return "No prior VASP step inputs available (this is the first VASP step)."
    # Keys that affect FFT grid, spin/SOC framework, or CHGCAR compatibility.
    # Planners are free to change other knobs, but these must be reasoned about
    # explicitly when chaining ICHARG>=10 or reusing WAVECAR.
    grid_keys = (
        "ENCUT", "PREC", "ISPIN", "LSORBIT", "LNONCOLLINEAR",
        "LHFCALC", "METAGGA", "GGA", "LDAU", "LMAXMIX",
    )
    blocks = []
    for entry in prior_step_inputs:
        step_name = entry.get("step_name", "?")
        incar = entry.get("incar", {}) or {}
        kpoints_summary = entry.get("kpoints_summary", "") or "(unknown)"
        shown = {k: incar[k] for k in grid_keys if k in incar}
        other_count = max(0, len(incar) - len(shown))
        lines = [f"## Step: {step_name}"]
        if shown:
            lines.append("  Grid/framework INCAR keys:")
            for k, v in shown.items():
                lines.append(f"    {k} = {v}")
            if other_count:
                lines.append(f"  (+ {other_count} other INCAR keys not shown)")
        else:
            lines.append("  (INCAR had none of the cross-step-sensitive keys explicitly set)")
        lines.append(f"  KPOINTS: {kpoints_summary}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


CROSS_STEP_CONSISTENCY_NOTES = """CROSS-STEP CONSISTENCY NOTES:
- When this step sets ICHARG >= 10 (e.g. 11 for NSCF band structures, 1 for
  restart), it reads the CHGCAR produced by an earlier step. VASP requires the
  FFT grid to match between the CHGCAR source and the current step, which means
  ENCUT and PREC must be identical to the source step. If you omit ENCUT or
  PREC, VASP derives them from POTCAR defaults and the grid may differ, causing
  `charge density could not be read from file CHGCAR for ICHARG>10`. When in
  doubt, copy ENCUT and PREC verbatim from the SCF step that wrote the CHGCAR.
- ISPIN, LSORBIT, LNONCOLLINEAR, LHFCALC, METAGGA, and LDAU/LDAUU must also
  stay consistent with the CHGCAR / WAVECAR source unless you are deliberately
  changing the electronic framework — doing so generally requires a fresh SCF
  (ICHARG=2) rather than an NSCF restart.
- For non-collinear (LSORBIT=.TRUE.) steps, a collinear WAVECAR cannot be
  reused (spinor basis doubles the coefficients). Start from the CHGCAR only
  (ISTART=0, ICHARG=1) unless a prior SOC step produced a compatible WAVECAR."""


def build_step_planner_prompt(
    skeletal_step: dict,
    material_context: dict,
    prior_outcomes: list[dict],
    prior_step_inputs: list[dict] | None = None,
) -> tuple[str, str]:
    user_prompt = f"""Knowledge base:
{VASP_KNOWLEDGE_BASE}

Material context:
{json.dumps(material_context, indent=2, ensure_ascii=False)}

Current skeletal step:
{json.dumps(skeletal_step, indent=2, ensure_ascii=False)}

Prior outcomes:
{format_step_outcomes_for_prompt(prior_outcomes)}

Prior step inputs (actual parameters that were written to disk):
{format_prior_step_inputs_for_prompt(prior_step_inputs or [])}

{CROSS_STEP_CONSISTENCY_NOTES}

Return JSON with keys:
- incar
- kpoints
- additional_files
- vasp_version
- reasoning_trace
- confidence
- fallback_parameters
"""
    return STEP_PLANNER_SYSTEM_PROMPT, user_prompt


def build_reasoning_monitor_prompt(monitor_context: dict) -> tuple[str, str]:
    user_prompt = f"""Analyze this running calculation state:
{json.dumps(monitor_context, indent=2, ensure_ascii=False)}

Return JSON with keys:
- decision
- confidence
- reasoning_trace
- should_kill
- suggested_action
"""
    return REASONING_MONITOR_SYSTEM_PROMPT, user_prompt


def build_reasoning_recovery_prompt(recovery_context: dict, validation_error: str | None = None) -> tuple[str, str]:
    error_section = ""
    if validation_error:
        error_section = f"\nPREVIOUS RESPONSE WAS INVALID: {validation_error}\nPlease correct your response.\n"
    user_prompt = f"""Knowledge base:
{VASP_KNOWLEDGE_BASE}

Recovery context:
{json.dumps(recovery_context, indent=2, ensure_ascii=False)}
{error_section}
Return JSON with keys:
- updated_parameters: dict with the following fields:
    - "incar" OR "incar_overrides" (mutually exclusive, exactly one must be provided):
        - "incar": full replacement INCAR dict (non-null, non-empty)
        - "incar_overrides": partial patch dict with only the keys to change
      Providing both, or neither (both null/missing), is an error.
    - "kpoints": full kpoints string/dict, or null to keep unchanged
    - "kpoints_overrides": partial kpoints patch, or null; mutually exclusive with "kpoints"
    - "vasp_version": string or null to keep unchanged
    - "confidence": float 0-1 or null
    - "additional_files": dict or null
- reasoning_trace: string explaining the diagnosis and fix
- should_modify_plan: bool
- plan_revision_note: string or null
"""
    return REASONING_RECOVERY_SYSTEM_PROMPT, user_prompt


def build_step_reflector_prompt(reflection_context: dict) -> tuple[str, str]:
    user_prompt = f"""Reflection context:
{json.dumps(reflection_context, indent=2, ensure_ascii=False)}

Return JSON with keys:
- decision: "accept" | "redo" | "modify_plan"
- reasoning_trace: string explaining your reasoning
- modified_plan: null if decision != "modify_plan"; otherwise a COMPLETE plan dict with ALL of the following keys (copy unchanged fields verbatim from current_skeletal_plan):
    {{
      "task_description": string,
      "material_formula": string,
      "material_type": string,
      "overall_strategy": string,
      "steps": [...],
      "global_considerations": [...],
      "reasoning_trace": string,
      "material_id": string,
      "metadata": {{}}
    }}
- next_step_index: int (index of next step to execute in the new or current plan)
"""
    return STEP_REFLECTOR_SYSTEM_PROMPT, user_prompt
