"""
Supported Properties Checker

This module defines which material properties the pipeline can currently calculate,
and provides functions to check if a given task description is supported.
"""
import logging
import re
from typing import Tuple, List, Optional


# Dictionary of supported properties with their keywords and descriptions
# 支持的属性字典（关键词和描述）
SUPPORTED_PROPERTIES = {
    "bandgap": {
        "keywords": ["bandgap", "band gap", "band-gap", "energy gap"],
        "description": "Electronic bandgap calculation (direct/indirect gap, VBM/CBM)",
        "extraction_available": True,
        "evaluation_available": True,
    },
    "band_structure": {
        "keywords": ["band structure", "band-structure", "electronic bands"],
        "description": "Electronic band structure along high-symmetry k-path",
        "extraction_available": True,
        "evaluation_available": False,  # No ground truth comparison
    },
    "dos": {
        "keywords": ["dos", "density of states", "pdos", "projected dos"],
        "description": "Density of states (total and projected)",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "space_group": {
        "keywords": ["space group", "space-group", "symmetry", "crystal symmetry"],
        "description": "Space group number and symbol from relaxed structure",
        "extraction_available": True,
        "evaluation_available": True,
    },
    "magnetic_moment": {
        "keywords": ["magnetic", "magnetization", "spin", "magnetic moment", "ferromagnetic", "antiferromagnetic"],
        "description": "Total magnetic moment from spin-polarized calculation",
        "extraction_available": True,
        "evaluation_available": True,
    },
    "heat_of_formation": {
        "keywords": ["heat of formation", "formation energy", "formation enthalpy"],
        "description": "Heat of formation per atom (requires elemental references)",
        "extraction_available": True,
        "evaluation_available": True,
    },
    "relaxation": {
        "keywords": ["relax", "optimization", "equilibrium", "geometry optimization"],
        "description": "Structure relaxation/optimization",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "total_energy": {
        "keywords": ["total energy", "dft energy", "ground state energy"],
        "description": "Total DFT energy of the system",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "raman": {
        "keywords": ["raman", "raman spectra", "raman active"],
        "description": "Raman spectra from DFPT calculation",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "phonon": {
        "keywords": ["phonon", "vibration", "vibrational", "phonon frequencies"],
        "description": "Phonon frequencies (at Gamma point via DFPT)",
        "extraction_available": True,
        "evaluation_available": False,
    },
    # 新增的支持属性 / Newly added supported properties
    "electronic_type": {
        "keywords": ["electronic type", "n-type", "p-type", "carrier type", 
                     "semiconductor type", "doping type", "intrinsic"],
        "description": "Electronic type classification (n-type, p-type, intrinsic, metal)",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "charged_defect": {
        "keywords": ["charged defect", "charge state", "nelect", "charged",
                     "defect charge", "vacancy charge", "ionized defect"],
        "description": "Charged defect calculations with NELECT adjustment",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "thermodynamics": {
        "keywords": ["thermodynamics", "entropy", "free energy", "helmholtz",
                     "gibbs", "heat capacity", "zpe", "zero point energy"],
        "description": "Thermodynamic properties from phonon calculations (S, F, Cv)",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "vdw_correction": {
        "keywords": ["vdw", "van der waals", "dft-d3", "dft-d", "dispersion",
                     "grimme", "ivdw", "d3 correction"],
        "description": "Van der Waals dispersion corrections (DFT-D3, TS, MBD)",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "dielectric": {
        "keywords": ["dielectric", "permittivity", "lepsilon", "born charges"],
        "description": "Dielectric tensor and Born effective charges",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "workfunction": {
        "keywords": ["work function", "workfunction", "surface potential"],
        "description": "Work function from slab calculations",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "neb": {
        "keywords": ["neb", "nudged elastic band", "transition state", "reaction path", 
                     "activation barrier", "energy barrier", "reaction barrier"],
        "description": "NEB transition state search and reaction path analysis",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "dielectric_function": {
        "keywords": ["dielectric function", "frequency dependent dielectric", "loptics",
                     "optical properties", "refractive index", "absorption coefficient"],
        "description": "Frequency-dependent dielectric function from LOPTICS",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "magnetic_comparison": {
        "keywords": ["magnetic comparison", "fm vs afm", "afm vs fm", "magnetic ordering",
                     "exchange coupling", "magnetic ground state"],
        "description": "Compare FM/AFM/NM configurations to find magnetic ground state",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "gw": {
        "keywords": ["gw", "g0w0", "quasiparticle", "gw calculation", "gw bandgap",
                     "qp energy", "gw0", "scgw"],
        "description": "GW quasiparticle calculations for accurate bandgaps",
        "extraction_available": True,
        "evaluation_available": False,
    },
    # 高级计算方法 / Advanced computational methods
    "hybrid_functional": {
        "keywords": ["hse", "hse06", "pbe0", "b3lyp", "hybrid functional", 
                     "hf exchange", "hartree-fock"],
        "description": "Hybrid functional calculations (HSE06, PBE0, B3LYP) for accurate electronic structure",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "dft_plus_u": {
        "keywords": ["dft+u", "dftu", "hubbard u", "ldau", "+u correction",
                     "gga+u", "lsda+u", "u parameter"],
        "description": "DFT+U calculations for strongly correlated systems",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "spin_orbit_coupling": {
        "keywords": ["soc", "spin-orbit", "spin orbit coupling", "lsorbit",
                     "spinor", "magnetic anisotropy", "mae"],
        "description": "Spin-orbit coupling calculations for heavy elements and MAE",
        "extraction_available": True,
        "evaluation_available": False,
    },
    # 表面和分子计算 / Surface and molecular calculations
    "surface": {
        "keywords": ["surface", "slab", "surface relaxation", "surface calculation",
                     "dipole correction", "vacuum layer"],
        "description": "Surface/slab calculations with proper vacuum and dipole corrections",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "molecule": {
        "keywords": ["molecule", "molecular", "gas phase", "isolated molecule",
                     "bond length", "dimer", "monomer"],
        "description": "Molecular/isolated atom calculations in a box",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "adsorption_energy": {
        "keywords": ["adsorption energy", "binding energy", "adsorbate binding",
                     "surface adsorption"],
        "description": "Adsorption energy calculations (requires manual setup)",
        "extraction_available": True,
        "evaluation_available": False,
    },
    "surface_energy": {
        "keywords": ["surface energy", "cleavage energy", "surface formation energy"],
        "description": "Surface energy calculations from slab models",
        "extraction_available": True,
        "evaluation_available": False,
    },
    # 分子动力学 / Molecular dynamics
    "md": {
        "keywords": ["molecular dynamics", "aimd", "ab initio md", "md simulation",
                     "nvt", "nve", "npt", "trajectory", "mdalgo"],
        "description": "Ab initio molecular dynamics simulations (NVE/NVT/NPT)",
        "extraction_available": True,
        "evaluation_available": False,
    },
    # 弹性性质 / Elastic properties
    "elastic": {
        "keywords": ["elastic", "elastic constants", "bulk modulus", "shear modulus",
                     "young modulus", "poisson ratio", "stiffness tensor"],
        "description": "Elastic constant calculations (workflow generation supported)",
        "extraction_available": False,
        "evaluation_available": False,
    },
    # IR 光谱 / IR spectra
    "ir_spectra": {
        "keywords": ["ir spectra", "infrared", "ir active", "dipole derivative"],
        "description": "Infrared (IR) spectra from DFPT calculation",
        "extraction_available": True,
        "evaluation_available": False,
    },
}


# Properties that are NOT fully supported (with explanations)
# 不完全支持的属性列表（包含原因说明）
UNSUPPORTED_PROPERTIES = {
    "energy_above_hull": {
        "keywords": ["energy above hull", "above hull", "convex hull", "hull energy", "thermodynamic stability"],
        "reason": "Requires access to materials database (e.g., Materials Project) for phase diagram construction. "
                  "The pipeline can calculate formation energy, but comparing against the convex hull requires "
                  "external thermodynamic data.",
        "workaround": "Use 'heat of formation' instead, then compare with external databases manually.",
    },
    "optical_advanced": {
        "keywords": ["bse", "bethe-salpeter", "exciton", "optical absorption spectra"],
        "reason": "Advanced optical property calculations (BSE, excitons) require additional VASP modules "
                  "and post-processing that are not fully implemented.",
        "workaround": "Basic dielectric function via LOPTICS is supported. Use 'dielectric function' or 'optical properties' keywords.",
    },
    "transport": {
        "keywords": ["transport", "conductivity", "seebeck", "thermal conductivity", "mobility", "boltzmann"],
        "reason": "Transport property calculations require BoltzTraP or similar codes "
                  "that are not integrated into the current pipeline.",
        "workaround": None,
    },
    "phonon_dispersion": {
        "keywords": ["phonon dispersion", "phonon band structure", "phonon dos"],
        "reason": "Full phonon dispersion calculations require supercell finite displacement method "
                  "with phonopy integration, which is not fully automated.",
        "workaround": "Gamma-point phonons via DFPT are supported. For full dispersion, manual phonopy setup required.",
    },
}


def check_task_support(task_description: str) -> Tuple[bool, List[str], List[dict]]:
    """
    Check if the task description requests properties that can be calculated.
    
    Args:
        task_description: The task description from config
        
    Returns:
        Tuple of:
        - is_supported: True if all requested properties are supported
        - matched_supported: List of supported property names that were matched
        - unsupported_matches: List of dicts with unsupported property info
    """
    task_lower = task_description.lower()
    
    matched_supported = []
    unsupported_matches = []
    
    # Check for supported properties
    for prop_name, prop_info in SUPPORTED_PROPERTIES.items():
        for keyword in prop_info["keywords"]:
            if keyword.lower() in task_lower:
                if prop_name not in matched_supported:
                    matched_supported.append(prop_name)
                break
    
    # Check for unsupported properties
    for prop_name, prop_info in UNSUPPORTED_PROPERTIES.items():
        for keyword in prop_info["keywords"]:
            if keyword.lower() in task_lower:
                unsupported_matches.append({
                    "property": prop_name,
                    "keyword_matched": keyword,
                    "reason": prop_info["reason"],
                    "workaround": prop_info.get("workaround"),
                })
                break
    
    is_supported = len(unsupported_matches) == 0
    
    return is_supported, matched_supported, unsupported_matches


def validate_task_and_warn(task_description: str) -> bool:
    """
    Validate the task description and print warnings/errors.
    
    This function should be called at the start of the pipeline to warn users
    about unsupported properties before any calculations begin.
    
    Args:
        task_description: The task description from config
        
    Returns:
        True if the task is supported and can proceed, False otherwise
    """
    is_supported, matched_supported, unsupported_matches = check_task_support(task_description)
    
    logging.info("="*70)
    logging.info("Task Support Validation")
    logging.info("="*70)
    logging.info(f"Task description: {task_description}")
    
    # Report matched supported properties
    if matched_supported:
        logging.info(f"\n[OK] Detected supported properties:")
        for prop in matched_supported:
            prop_info = SUPPORTED_PROPERTIES[prop]
            logging.info(f"  - {prop}: {prop_info['description']}")
            if prop_info.get("evaluation_available"):
                logging.info(f"    (Ground truth evaluation available)")
    
    # Report unsupported properties with warnings
    if unsupported_matches:
        logging.warning("\n" + "!"*70)
        logging.warning("WARNING: Unsupported properties detected!")
        logging.warning("!"*70)
        
        for match in unsupported_matches:
            logging.warning(f"\n[X] Property: {match['property']}")
            logging.warning(f"  Matched keyword: '{match['keyword_matched']}'")
            logging.warning(f"  Reason: {match['reason']}")
            if match.get("workaround"):
                logging.warning(f"  Workaround: {match['workaround']}")
        
        logging.warning("\n" + "!"*70)
        logging.warning("The pipeline may not produce valid results for these properties.")
        logging.warning("!"*70 + "\n")
        
        return False
    
    # No recognized properties at all
    if not matched_supported and not unsupported_matches:
        logging.warning("\n" + "?"*70)
        logging.warning("NOTE: Could not identify specific properties from task description.")
        logging.warning("?"*70)
        logging.warning("The LLM will attempt to generate a workflow based on the description.")
        logging.warning("If the task involves unsupported properties, results may be incomplete.")
        logging.warning("?"*70 + "\n")
        # Still allow to proceed - LLM might handle it
        return True
    
    logging.info("\n" + "="*70)
    logging.info("Task validation passed. Proceeding with workflow...")
    logging.info("="*70 + "\n")
    
    return True


def get_supported_properties_summary() -> str:
    """
    Get a formatted summary of all supported properties.
    
    Returns:
        Formatted string listing all supported properties
    """
    lines = ["Currently Supported Properties:", "="*50]
    
    for prop_name, prop_info in SUPPORTED_PROPERTIES.items():
        eval_status = "[Evaluable]" if prop_info.get("evaluation_available") else "[Compute only]"
        lines.append(f"\n{prop_name}:")
        lines.append(f"  Description: {prop_info['description']}")
        lines.append(f"  Keywords: {', '.join(prop_info['keywords'][:3])}...")
        lines.append(f"  Status: {eval_status}")
    
    lines.append("\n" + "="*50)
    lines.append("\nUnsupported Properties:")
    lines.append("-"*50)
    
    for prop_name in UNSUPPORTED_PROPERTIES.keys():
        lines.append(f"  - {prop_name}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print(get_supported_properties_summary())
    print("\n\n" + "="*70)
    print("Testing task validation:")
    print("="*70)
    
    test_tasks = [
        "Compute the bandgap for fast scanning",
        "Calculate energy above hull for stability analysis",
        "Determine the space group number from the relaxed structure",
        "Compute the elastic constants",
        "Calculate the total magnetic moment",
        "Run GW calculation for accurate bandgap",
        "Compute heat of formation and check stability",
    ]
    
    for task in test_tasks:
        print(f"\n--- Testing: {task} ---")
        validate_task_and_warn(task)

