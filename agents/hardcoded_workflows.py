"""
Hand-crafted workflow templates for the Rule-Based baseline.

With the "rule_based" configuration (CONSERVATIVE_MODE), the strategic planner
selects a template from the task description instead of calling an LLM. Tasks
that match no template fall back to the strategic planner's heuristic
relax / static / band / DOS template, executed with heuristic parameters.

Currently provided template:
- bandgap: relaxation -> static SCF on a dense k-mesh -> non-SCF band structure
  (ICHARG=11). Parameters follow the C2DB settings translated to VASP
  (https://2dhub.org/c2db/c2db.html): k-point density 6.0/A^-1 for relaxation,
  12.0/A^-1 for electronic properties, SIGMA = 0.05 eV.
"""

import logging
from typing import Optional, Dict, Any
from common_types.workflow_models import (
    WorkflowPlan, WorkflowStep, GlobalSettings, PostprocessingRules
)


logger = logging.getLogger(__name__)


def get_hardcoded_workflow(
    task_description: str,
    material_id: str,
    material_type: str = "auto"
) -> Optional[WorkflowPlan]:
    """
    Check if a hardcoded workflow exists for the given task.
    
    Args:
        task_description: User's task description
        material_id: Material identifier
        material_type: Material type (2D, 3D bulk, surface, molecule)
        
    Returns:
        WorkflowPlan if a hardcoded workflow matches, None otherwise
    """
    task_lower = task_description.lower()
    
    # Check for bandgap calculation
    bandgap_keywords = ["bandgap", "band gap", "band-gap", "带隙", "能带隙", 
                        "calculate bandgap", "compute bandgap", "计算带隙"]
    
    if any(keyword in task_lower for keyword in bandgap_keywords):
        logger.info(f"Found hardcoded workflow match: bandgap")
        return create_bandgap_workflow(material_id, material_type)
    
    return None


def create_bandgap_workflow(
    material_id: str,
    material_type: str = "auto"
) -> WorkflowPlan:
    """
    Create a hardcoded workflow for bandgap calculation.
    
    Standard workflow (3 steps):
    1. Structure relaxation (ISIF=2/3, k-density 6.0/Å⁻¹, LCHARG=.TRUE.)
    2. Static SCF with dense k-mesh (k-density 12.0/Å⁻¹, ICHARG=1, LCHARG=.TRUE.)
       → produces accurate, well-converged CHGCAR on dense uniform k-mesh
    3. Non-SCF band structure along k-path (ICHARG=11, reads CHGCAR from step 2)
    
    Why the intermediate static SCF step is needed:
    - Relaxation uses a coarse k-mesh (6.0/Å⁻¹) for computational efficiency
    - The resulting CHGCAR is too rough for accurate band structure eigenvalues
    - The static SCF step re-converges the charge density on a dense k-mesh
      (12.0/Å⁻¹), matching the C2DB "PBE electronic properties" standard
    - ICHARG=1 reads the relaxation CHGCAR as starting point → fast SCF convergence
    - The band structure step then reads this accurate CHGCAR via ICHARG=11
    
    Spin-polarization (ISPIN/MAGMOM/LMAXMIX) is NOT set here; the VASPExecutor
    fills in its default (ISPIN=2 for magnetic trigger elements or an odd
    electron count, otherwise VASP's ISPIN=1).
    
    CRITICAL: ICHARG=11 (non-SCF) is used for band structure:
    - ICHARG=2 does SCF on k-path, giving a poorly sampled charge density.
    - ICHARG=11 reads the well-converged CHGCAR from the static SCF step and
      computes eigenvalues non-self-consistently on the k-path. This is the
      standard VASP band structure approach.
    
    Parameters are based on C2DB database settings (GPAW → VASP translated):
    - Structure and energetics (C2DB steps 1-4): k-point density = 6.0/Å⁻¹,
      ENCUT = 800 eV, SIGMA = 0.05 eV, EDIFFG = -0.01 eV/Å,
      maximum stress = 0.002 eV/Å³
    - PBE electronic properties (C2DB steps 7-10): k-point density = 12.0/Å⁻¹
    - For 2D materials: vacuum = 15 Å (handled during POSCAR generation)
    
    Reference: https://2dhub.org/c2db/c2db.html
    
    Args:
        material_id: Material identifier
        material_type: Material type (2D, 3D bulk, surface, molecule)
        
    Returns:
        WorkflowPlan for bandgap calculation
    """
    logger.info(f"Creating hardcoded bandgap workflow for {material_id} (material_type={material_type})")
    
    # Determine k-point settings based on material type
    # C2DB uses k-point density, which vaspkit can convert to grid automatically
    if material_type == "2D":
        # 2D materials: C2DB uses k-point density = 6.0/Å⁻¹ for relaxation
        # vaspkit will convert this to appropriate grid (typically 12x12x1 or denser)
        relaxation_kpoints = "k-point density 6.0 per Å⁻¹"  # C2DB standard
        band_kpoints = "k-path"  # Will use vaspkit task 302 to generate k-path with density 12.0/Å⁻¹
    elif material_type == "molecule":
        # Molecules: Gamma-only
        relaxation_kpoints = "Gamma-only"
        band_kpoints = "Gamma-only"
    else:
        # 3D bulk: standard k-mesh (can use density or explicit grid)
        relaxation_kpoints = "k-point density 6.0 per Å⁻¹"  # C2DB standard
        band_kpoints = "k-path"  # Will use vaspkit task 303 to generate k-path
    
    # Step 1: Structure relaxation
    # C2DB parameters: k-point density = 6.0/Å⁻¹, ENCUT = 800 eV, SIGMA = 0.05 eV,
    # EDIFFG = -0.01 eV/Å, maximum stress = 0.002 eV/Å³
    #
    # ISIF for 2D materials: ISIF=2 (relax ions only, fix cell)
    #   2D materials have a vacuum layer in c-direction; ISIF=3 would
    #   incorrectly relax the vacuum, collapsing or expanding the slab.
    #   C2DB constrains the c-axis and only relaxes in-plane + ionic positions.
    # ISIF for 3D bulk: ISIF=3 (full cell + ion relaxation)
    relaxation_isif = 2 if material_type == "2D" else 3
    
    relaxation_step = WorkflowStep(
        name="01_relaxation",
        description=f"Structure relaxation (ISIF={relaxation_isif}) - C2DB parameters",
        input_files_needed=["POSCAR"],
        output_files_produced=["CONTCAR", "OUTCAR", "vasprun.xml", "CHGCAR"],
        INCAR={
            "NSW": 200,
            "ISIF": relaxation_isif,  # 2 for 2D (fix cell), 3 for 3D (full relax)
            "IBRION": 2,  # Conjugate gradient
            "EDIFF": 1e-6,  # Electronic convergence
            "EDIFFG": -0.01,  # Force convergence (eV/Å) - C2DB standard
            "PREC": "Accurate",
            "ENCUT": 800 if material_type == "2D" else 500,  # C2DB uses 800 eV for 2D materials
            # ISPIN, MAGMOM, LMAXMIX: auto-detected by VASPExecutor
            # based on POSCAR elements and POTCAR electron count
            "ISMEAR": 0,  # Gaussian smearing
            "SIGMA": 0.05,  # Fermi smearing (eV) - C2DB standard
            "SYMPREC": 1e-4,  # Looser symmetry tolerance for C2DB relaxed structures
                              # (C2DB CIFs have a≠b by ~0.0004Å, γ≠120° by ~0.003° → VASP
                              #  Bravais lattice mismatch at default SYMPREC=1e-5)
            "LREAL": "Auto",
            "LWAVE": ".FALSE.",
            "LCHARG": ".TRUE.",  # Need CHGCAR for static SCF step (ICHARG=1 reads as starting point)
        },
        KPOINTS=relaxation_kpoints,
        POTCAR_notes="Standard PBE pseudopotentials (C2DB uses PBE)",
        vaspkit_actions=[
            "Use vaspkit task 102 with k-point density 6.0/Å⁻¹ for 2D materials" if material_type == "2D" else ""
        ],
        sanity_checks=[
            "Check forces < 0.01 eV/Å (C2DB standard)",
            "Check stress < 0.002 eV/Å³ (C2DB standard)",
            "Check electronic convergence",
            "Check if NSW limit reached"
        ],
        expected_time_estimate="1-3 hours",
        depends_on=[],
        agent_type="computing"
    )
    
    # Step 2: Static SCF with dense k-mesh
    # Purpose: Produce an accurate, well-converged CHGCAR on a dense uniform k-mesh.
    # The relaxation step uses a coarse k-mesh (6.0/Å⁻¹) for efficiency, so its
    # CHGCAR is too rough for accurate band structure eigenvalues. This intermediate
    # step re-converges the charge density self-consistently on a denser k-mesh
    # (12.0/Å⁻¹), matching the C2DB "PBE electronic properties" stage (steps 7-10).
    #
    # C2DB → VASP parameter translation:
    # - C2DB k-point density 12.0/Å⁻¹ → vaspkit generates uniform k-mesh at this density
    # - C2DB PW cutoff 800 eV → ENCUT = 800
    # - C2DB Fermi smearing 0.05 eV → ISMEAR = 0, SIGMA = 0.05
    # - C2DB PBE functional → default VASP PBE (GGA = PE)
    # - ICHARG = 1: read relaxation CHGCAR as starting point, then do SCF
    #   (interpolates from coarse to dense grid automatically, converges fast)
    #
    # NOTE: C2DB uses GPAW, not VASP. The plane-wave cutoff and smearing are
    # directly transferable. The k-point density concept is the same but the
    # resulting grid dimensions differ slightly due to code-specific conventions.
    static_scf_kpoints = "k-point density 12.0 per Å⁻¹"  # C2DB PBE electronic properties standard
    
    static_scf_step = WorkflowStep(
        name="02_static_scf",
        description="Static SCF with dense k-mesh (12.0/Å⁻¹) for accurate CHGCAR - C2DB parameters",
        input_files_needed=["CONTCAR", "CHGCAR"],
        output_files_produced=["OUTCAR", "CHGCAR", "vasprun.xml"],
        INCAR={
            "NSW": 0,  # Static calculation (no ionic relaxation)
            "ISIF": 0,  # No cell relaxation or stress
            "IBRION": -1,  # No ionic movement
            "ICHARG": 1,  # Read CHGCAR from relaxation as starting point, then do SCF
                          # (much faster than ICHARG=2 since relaxation CHGCAR is close to ground state)
            "EDIFF": 1e-6,  # Tight electronic convergence for accurate charge density
            "PREC": "Accurate",
            "ENCUT": 800 if material_type == "2D" else 500,  # Same as relaxation (C2DB standard)
            # ISPIN, MAGMOM, LMAXMIX: auto-detected by VASPExecutor
            # (same detection as relaxation, ensuring consistency)
            "ISMEAR": 0,  # Gaussian smearing (C2DB uses Fermi smearing = 0.05 eV)
            "SIGMA": 0.05,  # C2DB standard
            "SYMPREC": 1e-4,  # Same as relaxation (C2DB relaxed structures need looser tolerance)
            "LREAL": ".FALSE.",  # Reciprocal space for accurate charge density
            "LWAVE": ".FALSE.",  # Don't need WAVECAR
            "LCHARG": ".TRUE.",  # CRITICAL: output accurate CHGCAR for band structure step
            "NELM": 200,  # Generous limit for tight SCF convergence
        },
        KPOINTS=static_scf_kpoints,
        POTCAR_notes="Same as relaxation step (PBE pseudopotentials)",
        vaspkit_actions=[
            "Use vaspkit task 102 with k-point density 12.0/Å⁻¹ to generate dense uniform k-mesh (C2DB PBE electronic properties standard)"
        ],
        sanity_checks=[
            "Check electronic convergence (EDIFF = 1e-6)",
            "Verify CHGCAR file is produced and non-empty",
            "Check total energy is reasonable compared to relaxation"
        ],
        expected_time_estimate="0.5-1 hours",
        depends_on=["01_relaxation"],
        agent_type="computing"
    )
    
    # Step 3: Non-SCF band structure along k-path
    # C2DB parameters: k-point density = 12.0/Å⁻¹ for PBE electronic properties
    #
    # IMPORTANT notes on VASP band structure:
    # - ICHARG=11: read the well-converged CHGCAR from the static SCF step (NOT from relaxation!)
    #   The static SCF CHGCAR is computed on a dense 12.0/Å⁻¹ k-mesh, giving accurate
    #   eigenvalues. Using the coarse relaxation CHGCAR (6.0/Å⁻¹) directly would give
    #   a poorly sampled charge density → inaccurate band structure.
    # - ISMEAR=0: Gaussian smearing (ISMEAR=-5 tetrahedron ONLY works with uniform k-mesh,
    #   NOT with k-path! Using -5 with k-path produces garbage results.)
    # - ISPIN: auto-detected by computing agent (must match previous steps for CHGCAR compatibility)
    # - LORBIT=11: project onto atomic orbitals for character analysis
    band_step = WorkflowStep(
        name="03_band_structure",
        description="Non-SCF band structure along k-path (ICHARG=11) using accurate CHGCAR from static SCF",
        input_files_needed=["CONTCAR", "CHGCAR"],
        output_files_produced=["EIGENVAL", "OUTCAR", "vasprun.xml", "BAND_GAP"],
        INCAR={
            "NSW": 0,  # Static calculation
            "ISIF": 0,  # No relaxation
            "IBRION": -1,  # No ionic relaxation
            "ICHARG": 11,  # Non-SCF: read well-converged CHGCAR from static SCF step
                           # (ICHARG=2 does SCF on k-path → poor charge density → wrong bandgap)
            "EDIFF": 1e-6,  # Electronic convergence
            "PREC": "Accurate",
            "ENCUT": 800 if material_type == "2D" else 500,  # C2DB uses 800 eV for 2D materials
            # ISPIN, MAGMOM, LMAXMIX: auto-detected by VASPExecutor
            # (same detection runs per-step, so all steps get consistent ISPIN)
            "ISMEAR": 0,  # Gaussian smearing (NOT -5! Tetrahedron is incompatible with k-path)
            "SIGMA": 0.05,  # Same as relaxation and static SCF
            "SYMPREC": 1e-4,  # Same as relaxation (C2DB relaxed structures need looser tolerance)
            "LREAL": ".FALSE.",  # Reciprocal space for accurate band structure
            "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE.",
            "LORBIT": 11,  # Projected band character
            "NELM": 120,  # Allow more electronic steps for convergence
        },
        KPOINTS=band_kpoints,
        POTCAR_notes="Same as relaxation step (PBE pseudopotentials)",
        vaspkit_actions=[
            f"Generate k-path using vaspkit task {'302' if material_type == '2D' else '303'} with k-point density 12.0/Å⁻¹ (C2DB standard)",
            "Extract bandgap and VBM/CBM using vaspkit"
        ],
        sanity_checks=[
            "Check electronic convergence",
            "Verify bandgap extraction"
        ],
        expected_time_estimate="0.5-2 hours",
        depends_on=["02_static_scf"],
        agent_type="computing"
    )
    
    # Global settings (C2DB parameters)
    # NOTE: ISPIN is intentionally left at default (1). The VASPExecutor
    # will auto-detect and set ISPIN=2 (with MAGMOM, LMAXMIX) only when needed
    # (magnetic elements or odd electron count). This avoids GPU OOM and ZHEGV
    # errors for non-magnetic materials like Ag8C4N8 and Cs2F2.
    global_settings = GlobalSettings(
        ENCUT=800 if material_type == "2D" else 500,  # C2DB uses 800 eV for 2D materials
        PREC="Accurate",
        EDIFF=1e-6,
        EDIFFG=-0.01,  # C2DB standard: maximum force = 0.01 eV/Å
        ISPIN=1,  # Default; auto-upgraded to 2 by computing agent when needed
        ALGO="Normal",
        LREAL="Auto",
        additional_settings={}
    )
    
    # Postprocessing rules
    postprocessing = PostprocessingRules(
        vaspkit_commands=[
            f"vaspkit -task {'302' if material_type == '2D' else '303'} to generate k-path with density 12.0/Å⁻¹ (C2DB standard)",
            "vaspkit band structure analysis to extract bandgap"
        ],
        extraction_rules={
            "bandgap": "Extract from BAND_GAP file or EIGENVAL",
            "vbm": "Valence band maximum energy",
            "cbm": "Conduction band minimum energy",
            "band_character": "Direct or indirect gap"
        },
        plotting_commands=[],
        output_format="json"
    )
    
    # Create workflow plan
    workflow_plan = WorkflowPlan(
        task_description=f"Calculate bandgap for {material_id} (C2DB parameters)",
        material_id=material_id,
        global_settings=global_settings,
        workflow_steps=[relaxation_step, static_scf_step, band_step],
        postprocessing=postprocessing,
        failure_modes=[],
        metadata={
            "workflow_type": "hardcoded",
            "workflow_name": "bandgap",
            "material_type": material_type,
            "parameters_source": "C2DB",
            "c2db_reference": "https://2dhub.org/c2db/c2db.html",
            "workflow_steps": "3 steps: relaxation → static SCF → non-SCF band structure",
            "relaxation_kpoint_density": "6.0/Å⁻¹",
            "static_scf_kpoint_density": "12.0/Å⁻¹",
            "band_kpoint_density": "12.0/Å⁻¹",
            "encut_eV": "800" if material_type == "2D" else "500",
            "sigma_eV": "0.05",
            "ediffg_eV_per_ang": "-0.01"
        },
        estimated_total_time="2-6 hours",
        required_resources={
            "nodes": 1,
            "cores_per_node": 8,
            "memory_gb": 16
        }
    )
    
    return workflow_plan
