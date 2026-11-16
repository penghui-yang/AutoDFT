"""
Work function extraction module.

Provides functions to:
- Extract work function from LOCPOT (planar average)
- Parse electrostatic potential data
- Calculate surface dipole corrections
"""
import os
import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WorkfunctionResult:
    """Container for work function calculation results."""
    # Work function (eV)
    work_function: float = 0.0
    
    # Fermi level relative to vacuum
    fermi_energy: float = 0.0
    
    # Vacuum level (electrostatic potential in vacuum region)
    vacuum_level: float = 0.0
    
    # For asymmetric slabs - two surfaces
    work_function_top: Optional[float] = None
    work_function_bottom: Optional[float] = None
    
    # Planar-averaged potential along z
    z_coordinates: Optional[np.ndarray] = None  # Å
    potential: Optional[np.ndarray] = None  # eV
    
    # Slab geometry info
    slab_center: float = 0.0  # z-position of slab center
    vacuum_start: float = 0.0  # z-position where vacuum starts
    vacuum_end: float = 0.0  # z-position where vacuum ends
    
    # Metadata
    is_symmetric: bool = True
    has_dipole_correction: bool = False


def extract_workfunction(
    calc_dir: str,
    fermi_from_outcar: bool = True
) -> Optional[WorkfunctionResult]:
    """
    Extract work function from VASP calculation.
    
    Requires LOCPOT file from calculation with LVHAR=.TRUE. or LVTOT=.TRUE.
    
    Args:
        calc_dir: Directory containing LOCPOT and OUTCAR
        fermi_from_outcar: Whether to read Fermi energy from OUTCAR
        
    Returns:
        WorkfunctionResult object or None
    """
    result = WorkfunctionResult()
    
    # Get Fermi energy from OUTCAR
    if fermi_from_outcar:
        outcar_path = os.path.join(calc_dir, 'OUTCAR')
        result.fermi_energy = _extract_fermi_energy(outcar_path)
    
    # Try vaspkit first (faster and more reliable)
    vaspkit_result = _extract_workfunction_vaspkit(calc_dir)
    if vaspkit_result is not None:
        result.work_function = vaspkit_result
        result.vacuum_level = result.fermi_energy + result.work_function
        logger.info(f"Work function from vaspkit: {result.work_function:.3f} eV")
        return result
    
    # Fall back to direct LOCPOT parsing
    locpot_path = os.path.join(calc_dir, 'LOCPOT')
    if not os.path.exists(locpot_path):
        logger.warning(f"LOCPOT not found in {calc_dir}")
        return None
    
    try:
        z_coords, potential = parse_locpot(locpot_path)
        result.z_coordinates = z_coords
        result.potential = potential
        
        # Find vacuum level
        result = _analyze_potential_profile(result)
        
        # Calculate work function
        result.work_function = result.vacuum_level - result.fermi_energy
        
        logger.info(f"Work function: {result.work_function:.3f} eV")
        logger.info(f"  Vacuum level: {result.vacuum_level:.3f} eV")
        logger.info(f"  Fermi energy: {result.fermi_energy:.3f} eV")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract work function: {e}")
        return None


def _extract_fermi_energy(outcar_path: str) -> float:
    """Extract Fermi energy from OUTCAR."""
    if not os.path.exists(outcar_path):
        logger.warning("OUTCAR not found for Fermi energy")
        return 0.0
    
    try:
        # Use reversed() to get the LAST E-fermi value (the converged one).
        # In relaxation OUTCARs (NSW>0), E-fermi appears once per ionic step;
        # the first is the initial (unconverged) value, the last is correct.
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "E-fermi" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "E-fermi":
                        return float(parts[i + 2])
    except Exception as e:
        logger.warning(f"Failed to extract Fermi energy: {e}")
    
    return 0.0


def _extract_workfunction_vaspkit(calc_dir: str) -> Optional[float]:
    """Use vaspkit to extract work function."""
    from execution.job_manager import run_command
    
    try:
        # Task 426: Calculate work function
        run_command("vaspkit -task 426", cwd=calc_dir, shell=True, timeout=60)
        
        # Parse output file
        wf_file = os.path.join(calc_dir, 'WORKFUNC.dat')
        if os.path.exists(wf_file):
            with open(wf_file, 'r') as f:
                for line in f:
                    if "Work function" in line or "WF" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.replace('.', '').replace('-', '').isdigit():
                                return float(part)
        
    except Exception as e:
        logger.debug(f"vaspkit work function extraction failed: {e}")
    
    return None


def parse_locpot(locpot_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse LOCPOT file and compute planar average along z.
    
    Args:
        locpot_path: Path to LOCPOT file
        
    Returns:
        Tuple of (z_coordinates, planar_averaged_potential)
    """
    with open(locpot_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    scale = float(lines[1].strip())
    
    # Lattice vectors
    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[2 + i].split()]
    lattice *= scale
    
    # Number of atoms
    atom_counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(atom_counts)
    
    # Find grid dimensions after atom positions
    # Skip: header (7 lines) + Selective dynamics (optional) + coordinate type + atoms
    line_idx = 7
    if lines[7].strip()[0].upper() == 'S':
        line_idx = 8
    line_idx += 1 + n_atoms  # Skip coordinate type and atoms
    
    # Skip empty line
    while line_idx < len(lines) and not lines[line_idx].strip():
        line_idx += 1
    
    # Grid dimensions
    ngx, ngy, ngz = [int(x) for x in lines[line_idx].split()]
    line_idx += 1
    
    # Read potential data
    potential_data = []
    values_per_line = 5  # VASP writes 5 values per line
    
    n_total = ngx * ngy * ngz
    while len(potential_data) < n_total and line_idx < len(lines):
        parts = lines[line_idx].split()
        potential_data.extend([float(x) for x in parts])
        line_idx += 1
    
    # Reshape to 3D grid
    potential_3d = np.array(potential_data[:n_total]).reshape((ngz, ngy, ngx))
    
    # Planar average along z
    potential_avg = np.mean(potential_3d, axis=(1, 2))
    
    # Z coordinates
    c_length = np.linalg.norm(lattice[2])
    z_coords = np.linspace(0, c_length, ngz, endpoint=False)
    
    return z_coords, potential_avg


def _analyze_potential_profile(result: WorkfunctionResult) -> WorkfunctionResult:
    """Analyze potential profile to find vacuum level and slab region."""
    if result.z_coordinates is None or result.potential is None:
        return result
    
    z = result.z_coordinates
    v = result.potential
    
    # Find vacuum region by looking for flat potential
    # Calculate gradient
    dv_dz = np.gradient(v, z)
    
    # Vacuum region has small gradient
    vacuum_threshold = 0.1  # eV/Å
    is_vacuum = np.abs(dv_dz) < vacuum_threshold
    
    # Find largest vacuum region
    # Group consecutive vacuum points
    vacuum_regions = []
    start = None
    for i, is_vac in enumerate(is_vacuum):
        if is_vac and start is None:
            start = i
        elif not is_vac and start is not None:
            vacuum_regions.append((start, i - 1))
            start = None
    if start is not None:
        vacuum_regions.append((start, len(is_vacuum) - 1))
    
    if vacuum_regions:
        # Find largest region
        largest = max(vacuum_regions, key=lambda x: x[1] - x[0])
        start_idx, end_idx = largest
        
        result.vacuum_start = z[start_idx]
        result.vacuum_end = z[end_idx]
        
        # Vacuum level is the average in vacuum region
        result.vacuum_level = np.mean(v[start_idx:end_idx + 1])
        
        # Slab center is the middle of non-vacuum region
        slab_indices = ~is_vacuum
        if np.any(slab_indices):
            slab_z = z[slab_indices]
            result.slab_center = (slab_z[0] + slab_z[-1]) / 2
    else:
        # No clear vacuum region found, use maximum potential
        result.vacuum_level = np.max(v)
        logger.warning("Could not identify clear vacuum region")
    
    return result


def calculate_dipole_correction(
    calc_dir: str,
    surface_normal: int = 3
) -> Dict[str, float]:
    """
    Calculate dipole correction for asymmetric slab.
    
    Args:
        calc_dir: Directory containing OUTCAR
        surface_normal: Direction normal to surface (1=x, 2=y, 3=z)
        
    Returns:
        Dictionary with dipole correction information
    """
    result = {
        'dipole_moment': 0.0,
        'potential_difference': 0.0,
        'dipole_correction_applied': False
    }
    
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar_path):
        return result
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Check if dipole correction was used
        if 'LDIPOL' in content and 'IDIPOL' in content:
            result['dipole_correction_applied'] = True
        
        # Extract dipole moment
        dipole_pattern = re.compile(r'dipole moment\s+p\[(x|y|z)\]=\s*([-\d.]+)')
        matches = dipole_pattern.findall(content)
        if matches:
            directions = {'x': 0, 'y': 1, 'z': 2}
            for direction, value in matches:
                if directions[direction] == surface_normal - 1:
                    result['dipole_moment'] = float(value)
        
        # Extract potential difference
        pot_diff_pattern = re.compile(r'total potential-Loss:\s*([-\d.]+)')
        match = pot_diff_pattern.search(content)
        if match:
            result['potential_difference'] = float(match.group(1))
        
    except Exception as e:
        logger.warning(f"Failed to extract dipole correction: {e}")
    
    return result
