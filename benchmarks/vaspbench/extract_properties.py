"""
Rule-based property extraction for VASPBench runs.

Parses the outputs of the last completed calculation of a workflow (energy,
structure, forces, symmetry, magnetic moments, eigenvalue gap, dielectric and
phonon data, depending on the calculation type).
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from tools.file_parsers import (
    extract_total_energy,
    extract_magnetic_moment,
    extract_space_group,
    get_formula_from_poscar,
    get_composition_from_poscar,
    extract_raman_spectra,
)

logger = logging.getLogger(__name__)


def parse_incar(incar_path: str) -> Dict[str, Any]:
    """
    Parse INCAR file to extract calculation parameters.
    
    Returns:
        Dictionary of INCAR parameters
    """
    params = {}
    if not os.path.exists(incar_path):
        return params
    
    try:
        with open(incar_path, 'r') as f:
            content = f.read()
        
        # Remove comments
        lines = []
        for line in content.split('\n'):
            # Remove comments (# or !)
            if '#' in line:
                line = line[:line.index('#')]
            if '!' in line:
                line = line[:line.index('!')]
            lines.append(line)
        content = '\n'.join(lines)
        
        # Parse key = value pairs
        # Pattern: KEY = VALUE (with optional semicolon)
        pattern = r'(\w+)\s*=\s*([^;\n]+)'
        for match in re.finditer(pattern, content, re.IGNORECASE):
            key = match.group(1).upper().strip()
            value = match.group(2).strip()
            
            # Try to convert to appropriate type
            try:
                if '.' in value or 'E' in value.upper():
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                # Keep as string if conversion fails
                # Handle .TRUE./.FALSE.
                if value.upper() in ['.TRUE.', 'TRUE', 'T', '.T.']:
                    params[key] = True
                elif value.upper() in ['.FALSE.', 'FALSE', 'F', '.F.']:
                    params[key] = False
                else:
                    params[key] = value
        
        return params
        
    except Exception as e:
        logger.warning(f"Failed to parse INCAR {incar_path}: {e}")
        return params


def detect_calculation_type(incar_params: Dict[str, Any]) -> str:
    """
    Detect the type of VASP calculation from INCAR parameters.
    
    Returns:
        Calculation type string
    """
    ibrion = incar_params.get('IBRION', -1)
    isif = incar_params.get('ISIF', 2)
    nsw = incar_params.get('NSW', 0)
    algo = str(incar_params.get('ALGO', 'Normal')).upper()
    loptics = incar_params.get('LOPTICS', False)
    lepsilon = incar_params.get('LEPSILON', False)
    lrpa = incar_params.get('LRPA', False)
    icorelevel = incar_params.get('ICORELEVEL', 0)
    lsorbit = incar_params.get('LSORBIT', False)
    
    # MD calculation
    if ibrion in [0] or str(incar_params.get('MDALGO', '')):
        return 'molecular_dynamics'
    
    # Phonon/DFPT calculation
    if ibrion in [5, 6, 7, 8]:
        return 'phonon_dfpt'
    
    # NEB calculation
    if 'IMAGES' in incar_params:
        return 'neb'
    
    # GW calculation
    if algo in ['GW', 'GW0', 'SCGW', 'SCGW0', 'G0W0']:
        return 'gw'
    
    # BSE calculation
    if algo == 'BSE':
        return 'bse'
    
    # RPA calculation
    if lrpa or algo == 'ACFDT':
        return 'rpa'
    
    # Dielectric/optical properties
    if loptics or lepsilon:
        return 'dielectric_optical'
    
    # XAS/XANES calculation
    if icorelevel > 0:
        return 'xas_xanes'
    
    # Relaxation calculation
    if nsw > 0 and ibrion in [1, 2, 3]:
        if isif >= 3:
            return 'volume_relaxation'
        else:
            return 'ionic_relaxation'
    
    # Spin-orbit coupling
    if lsorbit:
        return 'spin_orbit'
    
    # DOS calculation (typically ICHARG=11 or high NEDOS)
    icharg = incar_params.get('ICHARG', 0)
    nedos = incar_params.get('NEDOS', 301)
    if icharg == 11 or nedos > 1000:
        return 'dos'
    
    # Band structure (typically ICHARG=11 with line mode KPOINTS)
    # This is hard to detect from INCAR alone
    
    # Default: static/SCF calculation
    return 'static_scf'


def check_calculation_completed(calc_dir: str) -> Tuple[bool, str]:
    """
    Check if VASP calculation completed successfully.
    
    Returns:
        Tuple of (is_completed, status_message)
    """
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    
    if not os.path.exists(outcar_path):
        return False, "OUTCAR not found"
    
    try:
        # Check last lines of OUTCAR for completion indicators
        with open(outcar_path, 'rb') as f:
            # Read last 10KB
            f.seek(0, 2)  # Go to end
            file_size = f.tell()
            f.seek(max(0, file_size - 10000))
            content = f.read().decode('utf-8', errors='ignore')
        
        # Check for successful completion markers
        if "General timing and accounting informations for this job" in content:
            return True, "completed_normally"
        
        if "reached required accuracy" in content:
            return True, "converged"
        
        # Check for errors
        if "ZBRENT: fatal error" in content:
            return False, "zbrent_error"
        
        if "VERY BAD NEWS" in content:
            return False, "vasp_error"
        
        if "EDDDAV" in content and "error" in content.lower():
            return False, "electronic_convergence_error"
        
        # Check for ionic convergence
        if "TOTAL-FORCE" in content:
            return True, "force_calculation_done"
        
        # Partial completion
        return True, "partial_completion"
        
    except Exception as e:
        logger.warning(f"Error checking completion: {e}")
        return False, f"check_error: {e}"


def extract_lattice_parameters(contcar_path: str) -> Dict[str, Any]:
    """
    Extract lattice parameters from CONTCAR/POSCAR.
    
    Returns:
        Dictionary with lattice vectors, a/b/c, alpha/beta/gamma, volume
    """
    result = {
        'lattice_vectors': None,
        'a': None, 'b': None, 'c': None,
        'alpha': None, 'beta': None, 'gamma': None,
        'volume': None
    }
    
    if not os.path.exists(contcar_path):
        return result
    
    try:
        import numpy as np
        
        with open(contcar_path, 'r') as f:
            lines = f.readlines()
        
        # Line 1: scaling factor
        scale = float(lines[1].strip())
        
        # Lines 2-4: lattice vectors
        lattice = np.zeros((3, 3))
        for i in range(3):
            lattice[i] = [float(x) for x in lines[2 + i].split()[:3]]
        lattice *= scale
        
        # Calculate parameters
        a = np.linalg.norm(lattice[0])
        b = np.linalg.norm(lattice[1])
        c = np.linalg.norm(lattice[2])
        
        # Angles in degrees
        def angle(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        alpha = angle(lattice[1], lattice[2])
        beta = angle(lattice[0], lattice[2])
        gamma = angle(lattice[0], lattice[1])
        
        # Volume
        volume = abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
        
        result = {
            'lattice_vectors': lattice.tolist(),
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
            'volume': float(volume)
        }
        
    except Exception as e:
        logger.warning(f"Failed to extract lattice parameters from {contcar_path}: {e}")
    
    return result


def extract_forces(outcar_path: str) -> Dict[str, Any]:
    """
    Extract final forces from OUTCAR.
    
    Returns:
        Dictionary with max_force, forces array
    """
    result = {
        'max_force': None,
        'forces': None,
        'n_atoms': None
    }
    
    if not os.path.exists(outcar_path):
        return result
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Find last TOTAL-FORCE block
        # Pattern: POSITION                                       TOTAL-FORCE (eV/Angst)
        pattern = r'POSITION\s+TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\s*\n([\s\S]*?)\s*-+'
        matches = list(re.finditer(pattern, content))
        
        if matches:
            last_match = matches[-1]
            force_block = last_match.group(1)
            
            forces = []
            for line in force_block.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 6:
                    # x, y, z positions followed by fx, fy, fz forces
                    fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                    forces.append([fx, fy, fz])
            
            if forces:
                import numpy as np
                forces_arr = np.array(forces)
                max_force = np.max(np.abs(forces_arr))
                
                result = {
                    'max_force': float(max_force),
                    'forces': forces,
                    'n_atoms': len(forces)
                }
        
    except Exception as e:
        logger.warning(f"Failed to extract forces from {outcar_path}: {e}")
    
    return result


def extract_bandgap_from_eigenval(calc_dir: str) -> Dict[str, Any]:
    """
    Extract bandgap from EIGENVAL file without vaspkit.
    
    Returns:
        Dictionary with bandgap, vbm, cbm, is_metallic
    """
    result = {
        'bandgap_ev': None,
        'vbm_ev': None,
        'cbm_ev': None,
        'is_metallic': None,
        'fermi_energy': None
    }
    
    eigenval_path = os.path.join(calc_dir, 'EIGENVAL')
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    
    if not os.path.exists(eigenval_path):
        return result
    
    try:
        # Get Fermi energy from OUTCAR
        fermi_energy = None
        if os.path.exists(outcar_path):
            with open(outcar_path, 'r') as f:
                for line in f:
                    if 'E-fermi' in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == 'E-fermi':
                                fermi_energy = float(parts[i + 2])
        
        # Parse EIGENVAL
        with open(eigenval_path, 'r') as f:
            lines = f.readlines()
        
        # Line 6: NELECT, NKPT, NBANDS
        nelect, nkpt, nbands = [int(x) for x in lines[5].split()[:3]]
        
        # Determine number of electrons and spin
        ispin = 1
        if len(lines[5].split()) > 3:
            ispin = int(lines[5].split()[3]) if int(lines[5].split()[3]) in [1, 2] else 1
        
        # Parse eigenvalues
        all_eigenvalues = []
        line_idx = 7
        
        for k in range(nkpt):
            # Skip k-point header line
            line_idx += 1
            k_eigenvalues = []
            
            for b in range(nbands):
                parts = lines[line_idx].split()
                # Format: band_index, eigenvalue, [occupation]
                if len(parts) >= 2:
                    eigenval = float(parts[1])
                    k_eigenvalues.append(eigenval)
                line_idx += 1
            
            all_eigenvalues.append(k_eigenvalues)
            line_idx += 1  # Skip empty line
        
        if not all_eigenvalues:
            return result
        
        import numpy as np
        eigenvalues = np.array(all_eigenvalues)  # Shape: (nkpt, nbands)
        
        # Determine VBM and CBM
        # Use Fermi energy if available, otherwise use NELECT/2
        if fermi_energy is not None:
            # Find bands below and above Fermi level
            vbm = None
            cbm = None
            
            for k in range(nkpt):
                for b, ev in enumerate(eigenvalues[k]):
                    if ev <= fermi_energy:
                        if vbm is None or ev > vbm:
                            vbm = ev
                    else:
                        if cbm is None or ev < cbm:
                            cbm = ev
            
            if vbm is not None and cbm is not None:
                bandgap = cbm - vbm
                if bandgap < 0.01:  # Metallic
                    result = {
                        'bandgap_ev': 0.0,
                        'vbm_ev': float(vbm),
                        'cbm_ev': float(cbm),
                        'is_metallic': True,
                        'fermi_energy': float(fermi_energy)
                    }
                else:
                    result = {
                        'bandgap_ev': float(bandgap),
                        'vbm_ev': float(vbm),
                        'cbm_ev': float(cbm),
                        'is_metallic': False,
                        'fermi_energy': float(fermi_energy)
                    }
        else:
            # Fallback: use NELECT/2 for VBM band index
            n_occ_bands = int(nelect / 2)
            vbm = np.max(eigenvalues[:, n_occ_bands - 1])
            cbm = np.min(eigenvalues[:, n_occ_bands]) if n_occ_bands < nbands else vbm
            
            bandgap = cbm - vbm
            result = {
                'bandgap_ev': float(max(0, bandgap)),
                'vbm_ev': float(vbm),
                'cbm_ev': float(cbm),
                'is_metallic': bandgap < 0.01,
                'fermi_energy': None
            }
        
    except Exception as e:
        logger.warning(f"Failed to extract bandgap from EIGENVAL: {e}")
    
    return result


def extract_dielectric(outcar_path: str) -> Dict[str, Any]:
    """
    Extract dielectric properties from OUTCAR.
    
    Returns:
        Dictionary with dielectric tensor components
    """
    result = {
        'epsilon_static': None,  # Static dielectric constant
        'epsilon_ionic': None,   # Ionic contribution
        'epsilon_electronic': None,  # Electronic contribution
        'dielectric_tensor': None
    }
    
    if not os.path.exists(outcar_path):
        return result
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Look for static dielectric tensor
        # Pattern: MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)
        pattern = r'MACROSCOPIC STATIC DIELECTRIC TENSOR.*?\n\s*-+\s*\n\s*([\d.\s-]+)\n\s*([\d.\s-]+)\n\s*([\d.\s-]+)'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            tensor = []
            for i in range(1, 4):
                row = [float(x) for x in match.group(i).split()[:3]]
                tensor.append(row)
            
            result['dielectric_tensor'] = tensor
            # Average of diagonal elements
            result['epsilon_static'] = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3
        
        # Look for ionic and electronic contributions
        # MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION
        ionic_pattern = r'IONIC CONTRIBUTION.*?\n\s*-+\s*\n\s*([\d.\s-]+)\n\s*([\d.\s-]+)\n\s*([\d.\s-]+)'
        ionic_match = re.search(ionic_pattern, content, re.IGNORECASE)
        if ionic_match:
            tensor = []
            for i in range(1, 4):
                row = [float(x) for x in ionic_match.group(i).split()[:3]]
                tensor.append(row)
            result['epsilon_ionic'] = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3
        
    except Exception as e:
        logger.warning(f"Failed to extract dielectric from {outcar_path}: {e}")
    
    return result


def extract_born_charges(outcar_path: str) -> Dict[str, Any]:
    """
    Extract Born effective charges from OUTCAR.
    
    Returns:
        Dictionary with Born charges per atom
    """
    result = {
        'born_charges': None,
        'n_atoms': None
    }
    
    if not os.path.exists(outcar_path):
        return result
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Pattern: BORN EFFECTIVE CHARGES
        if 'BORN EFFECTIVE CHARGES' not in content:
            return result
        
        # Find the Born charges section
        pattern = r'BORN EFFECTIVE CHARGES.*?ion\s+\d+\s*\n\s*([\d.\s-]+)\n\s*([\d.\s-]+)\n\s*([\d.\s-]+)'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        
        if matches:
            born_charges = []
            for match in matches:
                tensor = []
                for i in range(1, 4):
                    row = [float(x) for x in match.group(i).split()[:3]]
                    tensor.append(row)
                # Average of diagonal elements
                z_eff = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3
                born_charges.append({
                    'Z_eff': z_eff,
                    'tensor': tensor
                })
            
            result['born_charges'] = born_charges
            result['n_atoms'] = len(born_charges)
        
    except Exception as e:
        logger.warning(f"Failed to extract Born charges from {outcar_path}: {e}")
    
    return result


def extract_all_properties(calc_dir: str, example_name: str = None) -> Dict[str, Any]:
    """
    Extract all available properties from a completed calculation.
    
    Args:
        calc_dir: Path to calculation directory
        example_name: Optional example name for identification
        
    Returns:
        Dictionary with all extracted properties
    """
    result = {
        'example_name': example_name,
        'calc_dir': calc_dir,
        'extraction_time': datetime.now().isoformat(),
    }
    
    # Check if calculation completed
    is_completed, status = check_calculation_completed(calc_dir)
    result['calculation_status'] = status
    result['is_completed'] = is_completed
    
    # Parse INCAR to determine calculation type
    incar_path = os.path.join(calc_dir, 'INCAR')
    incar_params = parse_incar(incar_path)
    calc_type = detect_calculation_type(incar_params)
    result['calculation_type'] = calc_type
    result['incar_params'] = incar_params
    
    # Basic files
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    contcar_path = os.path.join(calc_dir, 'CONTCAR')
    poscar_path = os.path.join(calc_dir, 'POSCAR')
    
    # Use CONTCAR if available, otherwise POSCAR
    structure_path = contcar_path if os.path.exists(contcar_path) else poscar_path
    
    # === Always extract these properties ===
    
    # 1. Total energy
    result['total_energy_ev'] = extract_total_energy(outcar_path)
    
    # 2. Formula and composition
    if os.path.exists(structure_path):
        result['formula'] = get_formula_from_poscar(structure_path)
        result['composition'] = get_composition_from_poscar(structure_path)
    
    # 3. Lattice parameters
    result['lattice'] = extract_lattice_parameters(structure_path)
    
    # 4. Forces (if available)
    forces_data = extract_forces(outcar_path)
    result['forces'] = forces_data
    
    # === Type-specific extractions ===
    
    # Space group (for relaxation/static calculations)
    if calc_type in ['static_scf', 'ionic_relaxation', 'volume_relaxation', 'dos']:
        if os.path.exists(structure_path):
            result['space_group'] = extract_space_group(structure_path)
    
    # Magnetic moment (for spin-polarized calculations)
    ispin = incar_params.get('ISPIN', 1)
    if ispin == 2:
        result['magnetic'] = extract_magnetic_moment(outcar_path)
    
    # Bandgap (for electronic structure calculations)
    if calc_type in ['static_scf', 'dos', 'ionic_relaxation', 'volume_relaxation']:
        result['electronic'] = extract_bandgap_from_eigenval(calc_dir)
    
    # Dielectric properties
    if calc_type in ['dielectric_optical', 'phonon_dfpt']:
        result['dielectric'] = extract_dielectric(outcar_path)
        result['born_charges'] = extract_born_charges(outcar_path)
    
    # Phonon frequencies
    if calc_type == 'phonon_dfpt':
        material_id = example_name or os.path.basename(calc_dir)
        result['phonon'] = extract_raman_spectra(calc_dir, material_id)
    
    # For spin-orbit calculations
    if calc_type == 'spin_orbit':
        result['spin_orbit'] = {
            'lsorbit': incar_params.get('LSORBIT', False),
            'saxis': incar_params.get('SAXIS', None),
        }
    
    return result
