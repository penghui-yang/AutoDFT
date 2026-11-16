"""
Magnetic properties analysis module.

Provides functions to:
- Extract magnetic moments from calculations
- Compare different magnetic configurations (FM, AFM, NM)
- Calculate magnetic exchange coupling constants (J)
"""
import os
import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MagneticResult:
    """Container for magnetic calculation results."""
    # Total magnetic moment (μ_B)
    total_moment: float = 0.0
    
    # Per-atom magnetic moments
    atom_moments: List[float] = field(default_factory=list)
    
    # Orbital-resolved magnetic moments (if available)
    orbital_moments: Optional[Dict[int, Dict[str, float]]] = None
    
    # Configuration type (FM, AFM, NM, FiM)
    config_type: str = ""
    
    # Total energy (eV)
    total_energy: float = 0.0
    
    # Convergence info
    is_converged: bool = False


@dataclass
class MagneticComparisonResult:
    """Container for comparing multiple magnetic configurations."""
    # Configuration results
    configurations: Dict[str, MagneticResult] = field(default_factory=dict)
    
    # Ground state
    ground_state: str = ""
    ground_state_energy: float = 0.0
    
    # Energy differences (relative to ground state)
    energy_differences: Dict[str, float] = field(default_factory=dict)
    
    # Exchange coupling constant J (if calculable)
    exchange_j: Optional[float] = None  # meV or eV
    exchange_j_unit: str = "meV"
    
    # Nearest neighbor spin value (S)
    spin_value: float = 0.5


def extract_magnetic_moments(calc_dir: str) -> Optional[MagneticResult]:
    """
    Extract magnetic moments from OUTCAR.
    
    Args:
        calc_dir: Directory containing OUTCAR
        
    Returns:
        MagneticResult object or None
    """
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar_path):
        logger.warning(f"OUTCAR not found in {calc_dir}")
        return None
    
    result = MagneticResult()
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Extract total magnetic moment
        # Pattern: "number of electron   XXX.XXX magnetization   YYY.YYY"
        total_mag_pattern = r"number of electron\s+[\d.]+\s+magnetization\s+([-\d.]+)"
        matches = re.findall(total_mag_pattern, content)
        if matches:
            result.total_moment = float(matches[-1])  # Last occurrence
        
        # Extract per-atom magnetic moments from the magnetization table
        # Look for "magnetization (x)" table
        atom_moments = _extract_atom_magnetizations(content)
        if atom_moments:
            result.atom_moments = atom_moments
        
        # Extract total energy
        energy = _extract_energy(content)
        if energy is not None:
            result.total_energy = energy
        
        # Check convergence
        result.is_converged = "reached required accuracy" in content
        
        # Determine configuration type
        result.config_type = _determine_config_type(result.atom_moments, result.total_moment)
        
        logger.info(f"Extracted magnetic moments: total = {result.total_moment:.3f} μ_B, "
                   f"config = {result.config_type}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract magnetic moments: {e}")
        return None


def _extract_atom_magnetizations(content: str) -> List[float]:
    """Extract per-atom magnetizations from OUTCAR."""
    atom_moments = []
    
    # Look for the magnetization table at the end of calculation
    # Pattern: "magnetization (x)"
    sections = content.split("magnetization (x)")
    if len(sections) < 2:
        return atom_moments
    
    # Get last occurrence
    last_section = sections[-1]
    
    # Parse the table
    # Format: "    1   -0.008   -0.004   -0.004   -0.005    0.001    2.145"
    # The last column is the total magnetization per atom
    lines = last_section.split('\n')
    for line in lines[:200]:  # Limit search
        # Skip header lines
        if line.strip().startswith('#') or 'tot' in line.lower():
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                # First part should be atom index
                atom_idx = int(parts[0])
                # Last part is total moment
                moment = float(parts[-1])
                atom_moments.append(moment)
            except (ValueError, IndexError):
                continue
        
        # Stop at empty line or non-data line
        if line.strip() == '' or line.strip().startswith('-'):
            if atom_moments:
                break
    
    return atom_moments


def _extract_energy(content: str) -> Optional[float]:
    """Extract total energy from OUTCAR content."""
    # Look for "energy  without entropy" line
    pattern = r"energy  without entropy\s*=\s*([-\d.]+)"
    matches = re.findall(pattern, content)
    if matches:
        return float(matches[-1])
    
    # Alternative: TOTEN
    pattern = r"free  energy   TOTEN\s*=\s*([-\d.]+)"
    matches = re.findall(pattern, content)
    if matches:
        return float(matches[-1])
    
    return None


def _determine_config_type(atom_moments: List[float], total_moment: float) -> str:
    """
    Determine magnetic configuration type based on atomic moments.
    
    FM: Ferromagnetic - all moments same sign
    AFM: Antiferromagnetic - alternating signs, total ≈ 0
    FiM: Ferrimagnetic - different magnitudes, non-zero total
    NM: Non-magnetic - all moments ≈ 0
    """
    if not atom_moments:
        if abs(total_moment) < 0.01:
            return "NM"
        return "FM" if total_moment > 0 else "FM"
    
    # Filter out negligible moments (e.g., from O atoms in oxides)
    significant_moments = [m for m in atom_moments if abs(m) > 0.1]
    
    if not significant_moments:
        return "NM"
    
    # Check if all moments have the same sign
    positive = sum(1 for m in significant_moments if m > 0)
    negative = len(significant_moments) - positive
    
    if positive == 0 or negative == 0:
        return "FM"
    
    # Check for AFM (approximately zero total, alternating signs)
    if abs(total_moment) < 0.1:
        return "AFM"
    
    # Ferrimagnetic: different magnitudes with net moment
    return "FiM"


def compare_magnetic_configurations(
    config_results: Dict[str, MagneticResult]
) -> MagneticComparisonResult:
    """
    Compare multiple magnetic configurations to find ground state.
    
    Args:
        config_results: Dictionary mapping config name to MagneticResult
            e.g., {"FM": result_fm, "AFM": result_afm, "NM": result_nm}
    
    Returns:
        MagneticComparisonResult with comparison data
    """
    comparison = MagneticComparisonResult()
    comparison.configurations = config_results
    
    # Find ground state (lowest energy)
    energies = {name: r.total_energy for name, r in config_results.items()}
    
    if not energies:
        logger.warning("No magnetic configurations to compare")
        return comparison
    
    ground_state_name = min(energies, key=energies.get)
    ground_state_energy = energies[ground_state_name]
    
    comparison.ground_state = ground_state_name
    comparison.ground_state_energy = ground_state_energy
    
    # Calculate energy differences
    comparison.energy_differences = {
        name: (energy - ground_state_energy) * 1000  # Convert to meV
        for name, energy in energies.items()
    }
    
    logger.info(f"Magnetic ground state: {ground_state_name}")
    for name, delta_e in comparison.energy_differences.items():
        logger.info(f"  {name}: ΔE = {delta_e:.2f} meV")
    
    return comparison


def calculate_exchange_coupling(
    e_fm: float,
    e_afm: float,
    n_magnetic_atoms: int = 2,
    spin_value: float = 0.5,
    coordination: int = 1
) -> float:
    """
    Calculate Heisenberg exchange coupling constant J.
    
    For a simple pair of spins:
        E_FM - E_AFM = -4 * J * S^2 * z
    
    where z is coordination number.
    
    Negative J: Antiferromagnetic coupling (AFM ground state)
    Positive J: Ferromagnetic coupling (FM ground state)
    
    Args:
        e_fm: Total energy of FM configuration (eV)
        e_afm: Total energy of AFM configuration (eV)
        n_magnetic_atoms: Number of magnetic atoms in the cell
        spin_value: Spin quantum number (typically 0.5 for single electron)
        coordination: Number of nearest neighbors
    
    Returns:
        Exchange coupling J in meV
    """
    # Simple formula: J = (E_FM - E_AFM) / (4 * S^2 * n_pairs)
    # For a cell with n magnetic atoms in AFM arrangement,
    # there are n_magnetic_atoms / 2 pairs
    
    delta_e = e_fm - e_afm  # eV
    n_pairs = n_magnetic_atoms // 2
    
    if n_pairs == 0:
        n_pairs = 1
    
    j_value = delta_e / (4 * spin_value**2 * n_pairs * coordination)
    
    # Convert to meV
    j_meV = j_value * 1000
    
    logger.info(f"Exchange coupling J = {j_meV:.2f} meV")
    if j_meV < 0:
        logger.info("  (Negative J indicates AFM coupling)")
    else:
        logger.info("  (Positive J indicates FM coupling)")
    
    return j_meV


def generate_magmom_string(
    n_atoms: int,
    moment: float = 5.0,
    config: str = "FM",
    magnetic_indices: Optional[List[int]] = None
) -> str:
    """
    Generate MAGMOM string for INCAR.
    
    Args:
        n_atoms: Total number of atoms
        moment: Initial moment magnitude (μ_B)
        config: Configuration type ("FM", "AFM", "AFM-A", "AFM-C", "AFM-G")
        magnetic_indices: Indices of magnetic atoms (0-based). If None, all atoms.
    
    Returns:
        MAGMOM string for INCAR
    """
    if magnetic_indices is None:
        magnetic_indices = list(range(n_atoms))
    
    magmom = [0.0] * n_atoms
    
    if config == "FM":
        # All same sign
        for i in magnetic_indices:
            magmom[i] = moment
    
    elif config in ["AFM", "AFM-A"]:
        # Simple alternating: ↑↓↑↓...
        for i, idx in enumerate(magnetic_indices):
            magmom[idx] = moment if i % 2 == 0 else -moment
    
    elif config == "AFM-C":
        # Layer alternating (for layered structures)
        # Assumes magnetic atoms are grouped by layer
        n_per_layer = len(magnetic_indices) // 2
        for i, idx in enumerate(magnetic_indices):
            layer = i // n_per_layer
            magmom[idx] = moment if layer % 2 == 0 else -moment
    
    elif config == "AFM-G":
        # Checkerboard (for cubic structures)
        # ↑↓↓↑ pattern for 4 atoms
        pattern = [1, -1, -1, 1]
        for i, idx in enumerate(magnetic_indices):
            magmom[idx] = moment * pattern[i % len(pattern)]
    
    elif config == "NM":
        # Non-magnetic
        pass  # All zeros
    
    else:
        logger.warning(f"Unknown config type: {config}, defaulting to FM")
        for i in magnetic_indices:
            magmom[i] = moment
    
    # Format as MAGMOM string
    # Try to compress repeated values (e.g., "4*5.0" instead of "5.0 5.0 5.0 5.0")
    return _compress_magmom_string(magmom)


def _compress_magmom_string(magmom: List[float]) -> str:
    """Compress MAGMOM list into VASP format with multipliers."""
    if not magmom:
        return ""
    
    result = []
    current_val = magmom[0]
    count = 1
    
    for val in magmom[1:]:
        if abs(val - current_val) < 1e-6:
            count += 1
        else:
            if count > 1:
                result.append(f"{count}*{current_val:.1f}")
            else:
                result.append(f"{current_val:.1f}")
            current_val = val
            count = 1
    
    # Don't forget the last group
    if count > 1:
        result.append(f"{count}*{current_val:.1f}")
    else:
        result.append(f"{current_val:.1f}")
    
    return " ".join(result)


def identify_magnetic_atoms(
    species: List[str],
    counts: List[int]
) -> Tuple[List[int], List[str]]:
    """
    Identify magnetic atoms based on element type.
    
    Args:
        species: List of element symbols in POSCAR order
        counts: Number of each species
    
    Returns:
        Tuple of (magnetic atom indices, magnetic element symbols)
    """
    # Common magnetic elements
    magnetic_elements = {
        # 3d transition metals
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
        # 4d transition metals  
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
        # 5d transition metals
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
        # Lanthanides
        'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        # Actinides
        'U', 'Np', 'Pu', 'Am'
    }
    
    # Strongly magnetic elements (almost always need ISPIN=2)
    strongly_magnetic = {'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'Gd', 'Eu'}
    
    magnetic_indices = []
    magnetic_species = []
    atom_idx = 0
    
    for species_name, count in zip(species, counts):
        # Check if this species is potentially magnetic
        if species_name in magnetic_elements:
            magnetic_species.append(species_name)
            for _ in range(count):
                magnetic_indices.append(atom_idx)
                atom_idx += 1
        else:
            atom_idx += count
    
    return magnetic_indices, list(set(magnetic_species))
