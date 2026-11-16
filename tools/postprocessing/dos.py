"""
DOS (Density of States) extraction and analysis module.

Provides functions to extract total and projected DOS from VASP calculations.
"""
import os
import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DOSData:
    """Container for DOS data."""
    energy: np.ndarray  # Energy grid (eV)
    total_dos: np.ndarray  # Total DOS
    integrated_dos: Optional[np.ndarray] = None  # Integrated DOS
    spin_up_dos: Optional[np.ndarray] = None  # Spin-up DOS (for ISPIN=2)
    spin_down_dos: Optional[np.ndarray] = None  # Spin-down DOS (for ISPIN=2)
    fermi_energy: float = 0.0  # Fermi energy (eV)
    is_spin_polarized: bool = False
    nedos: int = 0  # Number of energy points
    
    # Projected DOS data (orbital-resolved)
    pdos: Dict[str, np.ndarray] = field(default_factory=dict)
    # Atom-resolved PDOS: {atom_index: {orbital: dos_array}}
    atom_pdos: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)


def extract_dos_data(
    calc_dir: str,
    shift_fermi: bool = True
) -> Optional[DOSData]:
    """
    Extract DOS data from DOSCAR file.
    
    Args:
        calc_dir: Directory containing DOSCAR
        shift_fermi: Whether to shift energy to Fermi level
        
    Returns:
        DOSData object or None if extraction fails
    """
    doscar_path = os.path.join(calc_dir, 'DOSCAR')
    if not os.path.exists(doscar_path):
        logger.warning(f"DOSCAR not found in {calc_dir}")
        return None
    
    try:
        with open(doscar_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header (line 6)
        header = lines[5].split()
        emax = float(header[0])
        emin = float(header[1])
        nedos = int(header[2])
        efermi = float(header[3])
        
        # Determine if spin-polarized by checking column count
        first_data_line = lines[6].split()
        ncols = len(first_data_line)
        is_spin_polarized = ncols >= 5  # E, DOS_up, DOS_down, INT_up, INT_down
        
        # Read total DOS
        energies = []
        total_dos = []
        spin_up = []
        spin_down = []
        integrated = []
        
        for i in range(nedos):
            parts = lines[6 + i].split()
            energy = float(parts[0])
            if shift_fermi:
                energy -= efermi
            energies.append(energy)
            
            if is_spin_polarized:
                spin_up.append(float(parts[1]))
                spin_down.append(float(parts[2]))
                total_dos.append(float(parts[1]) + float(parts[2]))
                if len(parts) >= 5:
                    integrated.append(float(parts[3]) + float(parts[4]))
            else:
                total_dos.append(float(parts[1]))
                if len(parts) >= 3:
                    integrated.append(float(parts[2]))
        
        result = DOSData(
            energy=np.array(energies),
            total_dos=np.array(total_dos),
            fermi_energy=efermi if not shift_fermi else 0.0,
            is_spin_polarized=is_spin_polarized,
            nedos=nedos
        )
        
        if integrated:
            result.integrated_dos = np.array(integrated)
        if is_spin_polarized:
            result.spin_up_dos = np.array(spin_up)
            result.spin_down_dos = np.array(spin_down)
        
        logger.info(f"Extracted DOS with {nedos} points, E_F = {efermi:.4f} eV")
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract DOS: {e}")
        return None


def extract_pdos_data(
    calc_dir: str,
    atom_indices: Optional[List[int]] = None,
    shift_fermi: bool = True
) -> Optional[DOSData]:
    """
    Extract projected DOS (PDOS) data from DOSCAR.
    
    Requires LORBIT=11 or 12 in VASP calculation.
    
    Args:
        calc_dir: Directory containing DOSCAR
        atom_indices: List of atom indices to extract (0-based), None for all
        shift_fermi: Whether to shift energy to Fermi level
        
    Returns:
        DOSData object with PDOS information
    """
    doscar_path = os.path.join(calc_dir, 'DOSCAR')
    if not os.path.exists(doscar_path):
        logger.warning(f"DOSCAR not found in {calc_dir}")
        return None
    
    try:
        with open(doscar_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        natoms = int(lines[0].split()[0])
        header = lines[5].split()
        nedos = int(header[2])
        efermi = float(header[3])
        
        # First, extract total DOS
        result = extract_dos_data(calc_dir, shift_fermi)
        if result is None:
            return None
        
        # Read PDOS for each atom
        line_offset = 6 + nedos  # After total DOS section
        
        # Determine orbital columns from first atom's PDOS
        if line_offset < len(lines) - 1:
            first_pdos_line = lines[line_offset + 1].split()
            n_orbital_cols = len(first_pdos_line) - 1  # Exclude energy column
            
            # Determine orbital names based on column count
            # For LORBIT=11: s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2
            if n_orbital_cols >= 9:
                orbital_names = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']
            elif n_orbital_cols >= 4:
                orbital_names = ['s', 'p', 'd', 'f'][:n_orbital_cols]
            else:
                orbital_names = [f'orb{i}' for i in range(n_orbital_cols)]
        
        result.atom_pdos = {}
        
        for iatom in range(natoms):
            if atom_indices is not None and iatom not in atom_indices:
                line_offset += nedos + 1
                continue
            
            # Skip header line for this atom
            line_offset += 1
            
            atom_orbitals = {name: [] for name in orbital_names}
            
            for idos in range(nedos):
                parts = lines[line_offset + idos].split()
                for icol, orbital in enumerate(orbital_names):
                    if icol + 1 < len(parts):
                        atom_orbitals[orbital].append(float(parts[icol + 1]))
            
            result.atom_pdos[iatom] = {
                name: np.array(data) for name, data in atom_orbitals.items()
            }
            
            line_offset += nedos
        
        logger.info(f"Extracted PDOS for {len(result.atom_pdos)} atoms")
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract PDOS: {e}")
        return None


def extract_dos_from_vaspkit(calc_dir: str, task: int = 111) -> Dict[str, Any]:
    """
    Run vaspkit to extract DOS data.
    
    Args:
        calc_dir: Directory containing VASP outputs
        task: VASPKIT task number (111 for total DOS, 112 for PDOS)
        
    Returns:
        Dictionary with DOS information
    """
    from execution.job_manager import run_command
    
    try:
        run_command(f"vaspkit -task {task}", cwd=calc_dir, shell=True)
        
        result = {
            'vaspkit_task': task,
            'dos_extracted': True
        }
        
        # Check for output files
        tdos_file = os.path.join(calc_dir, 'TDOS.dat')
        if os.path.exists(tdos_file):
            result['tdos_file'] = tdos_file
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to run vaspkit DOS extraction: {e}")
        return {'dos_extracted': False, 'error': str(e)}
