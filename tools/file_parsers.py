"""
File parsing utilities for extracting information from VASP files.
"""
import logging
import os
import json
import re
from collections import deque
from typing import Optional, Dict, Any, List, Tuple


def get_formula_from_poscar(poscar_path: str) -> str:
    """
    Extract chemical formula from POSCAR file.
    
    Args:
        poscar_path: Path to the POSCAR file
        
    Returns:
        Chemical formula string (e.g., "C2H4")
    """
    try:
        with open(poscar_path, 'r') as f:
            lines = f.readlines()
            elements = lines[5].split()
            counts = lines[6].split()
            formula = "".join([f"{el}{c}" for el, c in zip(elements, counts)])
            return formula
    except Exception as e:
        logging.warning(f"Could not parse formula from {poscar_path}: {e}")
        return "Unknown"


def parse_bandgap_from_vaspkit(calc_dir: str, material_id: str) -> Dict[str, Any]:
    """
    Run vaspkit to analyze band gap and parse results.
    
    Handles two VASPKIT output formats:
    1. Non-spin-polarized: Has "Band Character: Direct/Indirect" line
    2. Spin-polarized (ISPIN=2): Has "Spin Channel: <UP> <DOWN> <TOTAL>" format
       without explicit Band Character line
    
    For metallic cases, distinguishes between Metal and Semi-metal based on
    whether VBM and CBM are at the same k-point location.
    
    Args:
        calc_dir: Directory containing VASP calculation outputs
        material_id: Material identifier
        
    Returns:
        Dictionary with id, calculated_character, and calculated_gap
    """
    from execution.job_manager import run_command
    
    logging.info(f"Analyzing bandgap for {material_id} in {calc_dir}")
    try:
        # vaspkit task 211 (Band-Structure -> Extract Band Data)
        # This creates a BAND_GAP file in the calc_dir
        run_command("vaspkit -task 211", cwd=calc_dir, shell=True)
        
        # Parse the BAND_GAP file
        band_gap_file = os.path.join(calc_dir, 'BAND_GAP')
        if not os.path.exists(band_gap_file):
            logging.warning(f"BAND_GAP file not found for {material_id}")
            return {
                'id': material_id,
                'calculated_character': "ExtractionFailed",
                'calculated_gap': None,
                'extraction_error': "BAND_GAP file not found"
            }
        
        band_gap = None
        band_character = None
        vbm_location = None
        cbm_location = None
        is_spin_polarized = False
        
        with open(band_gap_file, 'r') as f:
            for line in f:
                # Check if this is spin-polarized output
                if "Spin Channel:" in line:
                    is_spin_polarized = True
                
                if "Band Character:" in line:
                    band_character = line.split()[-1]
                elif "Band Gap (eV):" in line:
                    # For spin-polarized, format is: "Band Gap (eV):    1.06    1.06    1.06"
                    # Take the last value (TOTAL)
                    band_gap = float(line.split()[-1])
                
                # Parse VBM/CBM locations
                # Non-spin-polarized: "Location of VBM: x y z"
                # Spin-polarized: "Location of VBM (TO): x y z" (TOTAL row)
                if is_spin_polarized:
                    # For spin-polarized, use the (TO) = TOTAL row
                    if "Location of VBM (TO):" in line:
                        parts = line.split(':')[-1].strip().split()
                        vbm_location = tuple(float(x) for x in parts[:3])
                    elif "Location of CBM (TO):" in line:
                        parts = line.split(':')[-1].strip().split()
                        cbm_location = tuple(float(x) for x in parts[:3])
                else:
                    # Non-spin-polarized format
                    if "Location of VBM:" in line and "(UP)" not in line and "(DW)" not in line and "(TO)" not in line:
                        parts = line.split(':')[-1].strip().split()
                        vbm_location = tuple(float(x) for x in parts[:3])
                    elif "Location of CBM:" in line and "(UP)" not in line and "(DW)" not in line and "(TO)" not in line:
                        parts = line.split(':')[-1].strip().split()
                        cbm_location = tuple(float(x) for x in parts[:3])
        
        # For spin-polarized calculations without Band Character line,
        # determine Direct/Indirect based on VBM/CBM locations
        if band_gap is not None and band_character is None and is_spin_polarized:
            if vbm_location is not None and cbm_location is not None:
                tolerance = 1e-4  # Slightly larger tolerance for k-point comparison
                same_location = all(abs(v - c) < tolerance 
                                   for v, c in zip(vbm_location, cbm_location))
                
                # Check if it's metallic (very small gap)
                if band_gap < 0.01:  # Essentially zero gap
                    band_character = "Metallic"
                    logging.info(f"{material_id}: Spin-polarized with tiny gap ({band_gap:.4f} eV) -> Metallic")
                elif same_location:
                    band_character = "Direct"
                    logging.info(f"{material_id}: Spin-polarized, VBM=CBM location -> Direct")
                else:
                    band_character = "Indirect"
                    logging.info(f"{material_id}: Spin-polarized, VBM≠CBM location -> Indirect")
            else:
                logging.warning(f"{material_id}: Spin-polarized but could not parse VBM/CBM locations")
        
        if band_gap is not None and band_character is not None:
            # Normalize band character variations from different VASPKIT versions
            # "Semimetallic" should be treated the same as "Metallic" for Metal/Semi-metal classification
            if band_character.lower() in ["semimetallic", "semi-metallic"]:
                logging.info(f"{material_id}: Normalizing '{band_character}' to 'Metallic' for classification")
                band_character = "Metallic"
            
            # For metallic systems, distinguish Metal vs Semi-metal
            if band_character == "Metallic":
                if vbm_location is not None and cbm_location is not None:
                    # Check if VBM and CBM are at the same k-point
                    # Use a small tolerance for floating point comparison
                    tolerance = 1e-6
                    same_location = all(abs(v - c) < tolerance 
                                       for v, c in zip(vbm_location, cbm_location))
                    
                    if same_location:
                        band_character = "Metal"
                        logging.info(f"{material_id}: VBM and CBM overlap -> Metal")
                    else:
                        band_character = "Semi-metal"
                        logging.info(f"{material_id}: VBM and CBM at different k-points -> Semi-metal")
                else:
                    # If we can't determine locations, default to Metal
                    band_character = "Metal"
                    logging.warning(f"{material_id}: Could not parse VBM/CBM locations, defaulting to Metal")
            
            logging.info(f"Result for {material_id}: {band_character}, {band_gap} eV")
            return {
                'id': material_id,
                'calculated_character': band_character,
                'calculated_gap': band_gap
            }
        
        logging.warning(f"Could not parse bandgap from BAND_GAP file for {material_id}")
        return {
            'id': material_id,
            'calculated_character': "ExtractionFailed",
            'calculated_gap': None,
            'extraction_error': "Could not parse BAND_GAP file"
        }
    except Exception as e:
        logging.error(f"Failed to analyze bandgap for {material_id}: {e}")
        return {
            'id': material_id,
            'calculated_character': "ExtractionFailed",
            'calculated_gap': None,
            'extraction_error': str(e)
        }


def extract_space_group(poscar_path: str) -> Dict[str, Any]:
    """
    Extract space group information from POSCAR/CONTCAR using spglib.
    
    Args:
        poscar_path: Path to the POSCAR or CONTCAR file
        
    Returns:
        Dictionary with space_group_number, space_group_symbol, and has_inversion_symmetry
    """
    try:
        import spglib
    except ImportError:
        logging.warning("spglib not installed. Cannot extract space group.")
        return {
            'space_group_number': None,
            'space_group_symbol': None,
            'has_inversion_symmetry': None
        }
    
    try:
        # Parse POSCAR to get lattice, positions, and atomic numbers
        cell = _parse_poscar_for_spglib(poscar_path)
        if cell is None:
            return {
                'space_group_number': None,
                'space_group_symbol': None,
                'has_inversion_symmetry': None
            }
        
        # Get space group using spglib
        # Use a reasonable symmetry tolerance (0.01 Angstrom is typical)
        spacegroup = spglib.get_spacegroup(cell, symprec=0.01)
        
        if spacegroup is None:
            logging.warning(f"spglib could not determine space group for {poscar_path}")
            return {
                'space_group_number': None,
                'space_group_symbol': None,
                'has_inversion_symmetry': None
            }
        
        # Parse the spacegroup string (format: "Symbol (Number)")
        # e.g., "Fm-3m (225)" or "P6_3/mmc (194)"
        match = re.match(r'(.+?)\s*\((\d+)\)', spacegroup)
        if match:
            symbol = match.group(1).strip()
            number = int(match.group(2))
        else:
            symbol = spacegroup
            number = None
        
        # Determine inversion symmetry
        # Space groups with inversion symmetry have certain characteristics
        # A simpler approach: use spglib to get symmetry operations
        symmetry = spglib.get_symmetry(cell, symprec=0.01)
        has_inversion = False
        if symmetry is not None:
            rotations = symmetry['rotations']
            # Inversion operation is represented by -I (negative identity matrix)
            import numpy as np
            inversion_matrix = -np.eye(3, dtype=int)
            for rot in rotations:
                if np.allclose(rot, inversion_matrix):
                    has_inversion = True
                    break
        
        logging.info(f"Space group for {poscar_path}: {symbol} ({number}), inversion: {has_inversion}")
        
        return {
            'space_group_number': number,
            'space_group_symbol': symbol,
            'has_inversion_symmetry': has_inversion
        }
        
    except Exception as e:
        logging.error(f"Failed to extract space group from {poscar_path}: {e}")
        return {
            'space_group_number': None,
            'space_group_symbol': None,
            'has_inversion_symmetry': None
        }


def _parse_poscar_for_spglib(poscar_path: str) -> Optional[Tuple]:
    """
    Parse POSCAR file and return cell tuple for spglib.
    
    Args:
        poscar_path: Path to POSCAR file
        
    Returns:
        Tuple of (lattice, positions, numbers) for spglib, or None if parsing fails
    """
    import numpy as np
    
    # Element to atomic number mapping (common elements in materials science)
    ELEMENT_TO_Z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94
    }
    
    try:
        with open(poscar_path, 'r') as f:
            lines = f.readlines()
        
        # Line 0: comment
        # Line 1: scaling factor
        scale = float(lines[1].strip())
        
        # Lines 2-4: lattice vectors
        lattice = np.zeros((3, 3))
        for i in range(3):
            lattice[i] = [float(x) for x in lines[2 + i].split()[:3]]
        lattice *= scale
        
        # Line 5: element symbols
        elements = lines[5].split()
        
        # Line 6: number of each element
        counts = [int(x) for x in lines[6].split()]
        
        # Build atomic numbers array
        numbers = []
        for elem, count in zip(elements, counts):
            # Handle element symbols that might have extra characters
            elem_clean = elem.strip()
            if elem_clean in ELEMENT_TO_Z:
                z = ELEMENT_TO_Z[elem_clean]
            else:
                # Try to match first 1-2 characters
                if len(elem_clean) >= 2 and elem_clean[:2] in ELEMENT_TO_Z:
                    z = ELEMENT_TO_Z[elem_clean[:2]]
                elif elem_clean[0] in ELEMENT_TO_Z:
                    z = ELEMENT_TO_Z[elem_clean[0]]
                else:
                    logging.warning(f"Unknown element: {elem_clean}")
                    z = 0
            numbers.extend([z] * count)
        
        total_atoms = sum(counts)
        
        # Line 7: Selective dynamics (optional) or coordinate type
        line_idx = 7
        if lines[line_idx].strip()[0].upper() == 'S':
            line_idx = 8  # Skip selective dynamics line
        
        coord_type = lines[line_idx].strip()[0].upper()
        line_idx += 1
        
        # Read atomic positions
        positions = np.zeros((total_atoms, 3))
        for i in range(total_atoms):
            pos_line = lines[line_idx + i].split()[:3]
            positions[i] = [float(x) for x in pos_line]
        
        # Convert to fractional coordinates if in Cartesian
        if coord_type == 'C' or coord_type == 'K':
            # Cartesian coordinates - convert to fractional
            inv_lattice = np.linalg.inv(lattice)
            positions = np.dot(positions, inv_lattice)
        
        return (lattice, positions, numbers)
        
    except Exception as e:
        logging.error(f"Failed to parse POSCAR {poscar_path}: {e}")
        return None


def extract_magnetic_moment(outcar_path: str) -> Dict[str, Any]:
    """
    Extract total magnetic moment from OUTCAR file.
    
    Args:
        outcar_path: Path to the OUTCAR file
        
    Returns:
        Dictionary with total_magnetic_moment (in μ_B) and is_magnetic
    """
    result = {
        'total_magnetic_moment': None,
        'is_magnetic': None
    }
    
    if not os.path.exists(outcar_path):
        logging.warning(f"OUTCAR not found: {outcar_path}")
        return result
    
    try:
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        
        # Search for magnetization information from the end (get final value)
        # Look for "number of electron" line which contains spin information
        # Format: "number of electron     XXX.XXX magnetization     YYY.YYY"
        ispin = 1  # Default: non-spin-polarized
        total_mag = None
        
        for line in lines:
            # Check ISPIN setting
            if 'ISPIN' in line and '=' in line:
                parts = line.split('=')
                for i, part in enumerate(parts):
                    if 'ISPIN' in part and i + 1 < len(parts):
                        try:
                            ispin = int(parts[i + 1].split()[0])
                        except (ValueError, IndexError):
                            pass
        
        # If non-spin-polarized, magnetic moment is 0
        if ispin == 1:
            result['total_magnetic_moment'] = 0.0
            result['is_magnetic'] = False
            logging.info(f"ISPIN=1 (non-magnetic) in {outcar_path}")
            return result
        
        # For spin-polarized calculation, find the magnetization
        # Search from the end to get the final converged value
        for line in reversed(lines):
            if 'magnetization' in line.lower():
                # Try to extract the magnetization value
                # Format varies: "number of electron     X.XXX magnetization     Y.YYY"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() == 'magnetization' and i + 1 < len(parts):
                        try:
                            total_mag = float(parts[i + 1])
                            break
                        except ValueError:
                            continue
                if total_mag is not None:
                    break
        
        if total_mag is not None:
            result['total_magnetic_moment'] = abs(total_mag)  # Report absolute value
            # Consider material magnetic if |magnetization| > 0.1 μ_B
            result['is_magnetic'] = abs(total_mag) > 0.1
            logging.info(f"Total magnetic moment: {total_mag:.4f} μ_B from {outcar_path}")
        else:
            logging.warning(f"Could not extract magnetization from {outcar_path}")
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to extract magnetic moment from {outcar_path}: {e}")
        return result


def get_composition_from_poscar(poscar_path: str) -> Dict[str, int]:
    """
    Extract composition (element counts) from POSCAR file.
    
    Args:
        poscar_path: Path to the POSCAR file
        
    Returns:
        Dictionary mapping element symbols to counts, e.g., {"Si": 2, "O": 4}
    """
    try:
        with open(poscar_path, 'r') as f:
            lines = f.readlines()
        
        elements = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        
        composition = {}
        for elem, count in zip(elements, counts):
            composition[elem.strip()] = count
        
        return composition
        
    except Exception as e:
        logging.error(f"Failed to get composition from {poscar_path}: {e}")
        return {}


def load_elemental_references(reference_file: str = None) -> Dict[str, float]:
    """
    Load elemental reference energies from JSON file.
    
    Args:
        reference_file: Path to elemental_references.json. If None, uses default location.
        
    Returns:
        Dictionary mapping element symbols to reference energy per atom (eV/atom)
    """
    if reference_file is None:
        # Default location: data/elemental_references.json relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reference_file = os.path.join(base_dir, 'data', 'elemental_references.json')
    
    if not os.path.exists(reference_file):
        logging.warning(f"Elemental reference file not found: {reference_file}")
        return {}
    
    try:
        with open(reference_file, 'r') as f:
            data = json.load(f)
        
        # Extract energy_per_atom values
        references = {}
        for element, info in data.items():
            if isinstance(info, dict) and 'energy_per_atom' in info:
                references[element] = info['energy_per_atom']
            elif isinstance(info, (int, float)):
                references[element] = float(info)
        
        logging.info(f"Loaded elemental references for {len(references)} elements")
        return references
        
    except Exception as e:
        logging.error(f"Failed to load elemental references: {e}")
        return {}


def calculate_heat_of_formation(
    total_energy: float,
    composition: Dict[str, int],
    elemental_references: Dict[str, float] = None,
    reference_file: str = None
) -> Dict[str, Any]:
    """
    Calculate heat of formation from total energy and composition.
    
    Formula: ΔHf = (E_total - Σ n_i × μ_i) / N_atoms
    
    Args:
        total_energy: Total DFT energy of the compound (eV)
        composition: Dictionary mapping element symbols to counts
        elemental_references: Pre-loaded reference energies. If None, loads from file.
        reference_file: Path to reference file (only used if elemental_references is None)
        
    Returns:
        Dictionary with heat_of_formation (eV/atom) and related info
    """
    result = {
        'heat_of_formation': None,
        'total_energy': total_energy,
        'reference_energy_sum': None,
        'n_atoms': None
    }
    
    if total_energy is None:
        logging.warning("Total energy is None, cannot calculate heat of formation")
        return result
    
    if not composition:
        logging.warning("Empty composition, cannot calculate heat of formation")
        return result
    
    # Load elemental references if not provided
    if elemental_references is None:
        elemental_references = load_elemental_references(reference_file)
    
    if not elemental_references:
        logging.warning("No elemental references available, cannot calculate heat of formation")
        return result
    
    # Calculate reference energy sum and check for missing elements
    reference_sum = 0.0
    n_atoms = 0
    missing_elements = []
    
    for element, count in composition.items():
        n_atoms += count
        if element in elemental_references:
            reference_sum += count * elemental_references[element]
        else:
            missing_elements.append(element)
    
    if missing_elements:
        logging.warning(f"Missing elemental references for: {missing_elements}")
        return result
    
    # Calculate heat of formation per atom
    heat_of_formation = (total_energy - reference_sum) / n_atoms
    
    result['heat_of_formation'] = heat_of_formation
    result['reference_energy_sum'] = reference_sum
    result['n_atoms'] = n_atoms
    
    logging.info(f"Heat of formation: {heat_of_formation:.4f} eV/atom "
                 f"(E_total={total_energy:.4f}, E_ref={reference_sum:.4f}, N={n_atoms})")
    
    return result


def extract_total_energy(outcar_path: str) -> Optional[float]:
    """
    Extract final total energy from OUTCAR file.
    
    Args:
        outcar_path: Path to the OUTCAR file
        
    Returns:
        Total energy in eV, or None if extraction fails
    """
    if not os.path.exists(outcar_path):
        logging.warning(f"OUTCAR not found: {outcar_path}")
        return None
    
    try:
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        
        # Find last occurrence of "free  energy   TOTEN"
        # Format: "free  energy   TOTEN  =      -XXX.XXXXX eV"
        for line in reversed(lines):
            if "free  energy" in line and "TOTEN" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "TOTEN" and i + 2 < len(parts):
                        try:
                            energy = float(parts[i + 2])
                            logging.info(f"Total energy: {energy:.6f} eV from {outcar_path}")
                            return energy
                        except (IndexError, ValueError):
                            pass
                break
        
        logging.warning(f"Could not extract total energy from {outcar_path}")
        return None
        
    except Exception as e:
        logging.error(f"Failed to extract total energy from {outcar_path}: {e}")
        return None


def extract_max_force(outcar_path: str) -> Optional[float]:
    """Extract the maximum force on any atom from the last ionic step in OUTCAR."""
    if not os.path.exists(outcar_path):
        return None
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        # Find all TOTAL-FORCE blocks; use the last one (final ionic step)
        blocks = content.split("TOTAL-FORCE (eV/Angst)")
        if len(blocks) < 2:
            return None
        last_block = blocks[-1]
        max_f = 0.0
        for line in last_block.splitlines():
            line = line.strip()
            if line.startswith("-----------"):
                if max_f > 0:
                    break  # end of force table
                continue
            parts = line.split()
            if len(parts) == 6:
                try:
                    fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                    f_mag = (fx**2 + fy**2 + fz**2) ** 0.5
                    max_f = max(max_f, f_mag)
                except ValueError:
                    continue
        return max_f if max_f > 0 else None
    except Exception as e:
        logging.error(f"Failed to extract max force from {outcar_path}: {e}")
        return None


def extract_stress_tensor(outcar_path: str) -> Optional[List[List[float]]]:
    """Extract the stress tensor (in kB) from the last occurrence in OUTCAR.

    Returns a 3x3 matrix as list of lists, or None if not found.
    VASP prints: ``in kB   XX   YY   ZZ   XY   YZ   ZX``
    """
    if not os.path.exists(outcar_path):
        return None
    try:
        last_stress_line = None
        with open(outcar_path, 'r') as f:
            for line in f:
                if "in kB" in line:
                    last_stress_line = line
        if last_stress_line is None:
            return None
        parts = last_stress_line.split()
        # Expected: "in kB  XX YY ZZ XY YZ ZX"
        idx = parts.index("kB") + 1
        vals = [float(v) for v in parts[idx:idx + 6]]
        if len(vals) != 6:
            return None
        xx, yy, zz, xy, yz, zx = vals
        return [
            [xx, xy, zx],
            [xy, yy, yz],
            [zx, yz, zz],
        ]
    except Exception as e:
        logging.error(f"Failed to extract stress tensor from {outcar_path}: {e}")
        return None


def extract_cell_volume(poscar_path: str) -> Optional[float]:
    """Extract cell volume from a POSCAR/CONTCAR by computing det(lattice)."""
    if not os.path.exists(poscar_path):
        return None
    try:
        with open(poscar_path, 'r') as f:
            lines = f.readlines()
        scale = float(lines[1].strip())
        a = [float(x) for x in lines[2].split()]
        b = [float(x) for x in lines[3].split()]
        c = [float(x) for x in lines[4].split()]
        # Volume = |a . (b x c)| * scale^3
        cross = [
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ]
        vol = abs(a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) * abs(scale) ** 3
        return vol
    except Exception as e:
        logging.error(f"Failed to extract cell volume from {poscar_path}: {e}")
        return None


def extract_raman_spectra(calc_dir: str, material_id: str) -> Dict[str, Any]:
    """
    Extract phonon/Raman spectra data from VASP DFPT calculation.
    
    This function extracts phonon frequencies from OUTCAR after a DFPT calculation
    (IBRION=7 or 8 with LEPSILON=.TRUE.). It also checks for pre-existing RAMAN.dat
    and IR.dat files if available.
    
    Note: VASP DFPT calculates phonon frequencies. True Raman activities require
    additional calculations (second-order response). The frequencies extracted here
    are all phonon modes; symmetry analysis is needed to determine Raman/IR activity.
    
    Args:
        calc_dir: Directory containing VASP calculation outputs (OUTCAR)
        material_id: Material identifier
        
    Returns:
        Dictionary with:
            - raman_frequencies: List of phonon frequencies (cm^-1), excluding acoustic modes
            - raman_activities: List of Raman activities (if available, else placeholders)
            - ir_frequencies: List of IR-active mode frequencies (if available)
            - ir_intensities: List of IR intensities (if available)
            - n_raman_modes: Number of optical phonon modes
            - n_ir_modes: Number of IR-active modes (if determined)
            - has_raman_data: Boolean indicating if phonon data was extracted
    """
    result = {
        'raman_frequencies': [],
        'raman_activities': [],
        'ir_frequencies': [],
        'ir_intensities': [],
        'n_raman_modes': 0,
        'n_ir_modes': 0,
        'has_raman_data': False
        }


def extract_oszicar_tail(path: str, n: int = 30) -> str:
    """Return the last n lines of OSZICAR, or an empty string if unavailable."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r") as handle:
            return "".join(deque(handle, maxlen=n)).strip()
    except Exception as exc:
        logging.warning(f"Failed to read OSZICAR tail from {path}: {exc}")
        return ""


def extract_outcar_summary(path: str) -> str:
    """Extract a compact summary of warnings, timing, and final convergence markers."""
    if not os.path.exists(path):
        return ""

    warning_lines: List[str] = []
    timing_lines: List[str] = []
    final_lines = deque(maxlen=20)
    patterns = ("warning", "error", "elapsed time", "total cpu time", "reached required accuracy")

    try:
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                final_lines.append(line.rstrip())
                lower = line.lower()
                if "warning" in lower or "error" in lower:
                    warning_lines.append(line.rstrip())
                if "elapsed time" in lower or "total cpu time" in lower:
                    timing_lines.append(line.rstrip())
        selected = []
        selected.extend(warning_lines[-10:])
        selected.extend(timing_lines[-5:])
        selected.extend([line for line in final_lines if any(token in line.lower() for token in patterns)][-10:])
        return "\n".join(dict.fromkeys(selected)).strip()
    except Exception as exc:
        logging.warning(f"Failed to summarize OUTCAR {path}: {exc}")
        return ""


def extract_electronic_convergence(path: str) -> List[int]:
    """Parse approximate electronic step counts per ionic step from OSZICAR."""
    if not os.path.exists(path):
        return []
    counts: List[int] = []
    pattern = re.compile(r"^\s*(\d+)\s+F=\s")
    try:
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                match = pattern.match(line)
                if match:
                    counts.append(int(match.group(1)))
    except Exception as exc:
        logging.warning(f"Failed to parse electronic convergence from {path}: {exc}")
    return counts


def extract_convergence_profile(path: str) -> Dict[str, Any]:
    """Extract a lightweight convergence profile from OSZICAR."""
    profile = {"ionic_steps": 0, "energies": [], "energy_deltas": []}
    if not os.path.exists(path):
        return profile
    pattern = re.compile(r"^\s*(\d+)\s+F=\s*([-\d.E+]+)\s+E0=\s*([-\d.E+]+)\s+d\s*E\s*=\s*([-\d.E+]+)")
    try:
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                match = pattern.match(line)
                if match:
                    profile["ionic_steps"] += 1
                    profile["energies"].append(float(match.group(3)))
                    profile["energy_deltas"].append(float(match.group(4)))
    except Exception as exc:
        logging.warning(f"Failed to extract convergence profile from {path}: {exc}")
    return profile


def detect_symmetry_change(poscar_before: str, contcar_after: str) -> Optional[str]:
    """Compare pre/post relaxation symmetry labels if spglib is available."""
    before = extract_space_group(poscar_before)
    after = extract_space_group(contcar_after)
    before_symbol = before.get("space_group_symbol")
    after_symbol = after.get("space_group_symbol")
    if before_symbol is None or after_symbol is None:
        return None
    if before_symbol == after_symbol:
        return after_symbol
    return f"{before_symbol} -> {after_symbol}"


def estimate_bandgap_from_eigenval(path: str) -> Optional[float]:
    """Estimate the bandgap from EIGENVAL by finding the smallest occupied-unoccupied separation."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", errors="ignore") as handle:
            lines = handle.readlines()
        if len(lines) < 8:
            return None
        eigenvalues: List[Tuple[float, float]] = []
        for line in lines[8:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    eigenvalues.append((float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
        occupied = [ev for ev, occ in eigenvalues if occ > 0.5]
        unoccupied = [ev for ev, occ in eigenvalues if occ <= 0.5]
        if not occupied or not unoccupied:
            return 0.0
        return max(0.0, min(unoccupied) - max(occupied))
    except Exception as exc:
        logging.warning(f"Failed to estimate bandgap from {path}: {exc}")
        return None
    
    logging.info(f"Extracting phonon/Raman spectra for {material_id} in {calc_dir}")
    
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar_path):
        logging.warning(f"OUTCAR not found for {material_id}")
        return result
    
    try:
        # First check if pre-existing RAMAN.dat/IR.dat files exist
        # (could be from phonopy, vaspkit, or other post-processing)
        raman_file = os.path.join(calc_dir, 'RAMAN.dat')
        ir_file = os.path.join(calc_dir, 'IR.dat')
        
        # Parse RAMAN.dat if it exists
        if os.path.exists(raman_file):
            result = _parse_raman_file(raman_file, result)
            logging.info(f"Found {result['n_raman_modes']} modes from RAMAN.dat for {material_id}")
        
        # Parse IR.dat if it exists
        if os.path.exists(ir_file):
            result = _parse_ir_file(ir_file, result)
            logging.info(f"Found {result['n_ir_modes']} IR-active modes for {material_id}")
        
        # If no pre-existing files, parse phonon frequencies directly from OUTCAR
        if not result['has_raman_data']:
            result = _parse_raman_from_outcar(outcar_path, result)
        
        if result['raman_frequencies'] or result['ir_frequencies']:
            result['has_raman_data'] = True
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to extract phonon/Raman spectra for {material_id}: {e}")
        return result


def _parse_raman_file(raman_file: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse RAMAN.dat file generated by vaspkit.
    
    Expected format:
    # Frequency (cm^-1)   Raman Activity (A^4/amu)
    123.45               0.123
    ...
    
    Args:
        raman_file: Path to RAMAN.dat file
        result: Dictionary to update with parsed data
        
    Returns:
        Updated result dictionary
    """
    try:
        with open(raman_file, 'r') as f:
            lines = f.readlines()
        
        frequencies = []
        activities = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('--'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    activity = float(parts[1])
                    # Only include positive frequencies (exclude imaginary modes)
                    if freq > 0:
                        frequencies.append(freq)
                        activities.append(activity)
                except ValueError:
                    continue
        
        result['raman_frequencies'] = frequencies
        result['raman_activities'] = activities
        result['n_raman_modes'] = len(frequencies)
        if frequencies:
            result['has_raman_data'] = True
        
        return result
        
    except Exception as e:
        logging.warning(f"Failed to parse RAMAN.dat: {e}")
        return result


def _parse_ir_file(ir_file: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse IR.dat file generated by vaspkit.
    
    Expected format:
    # Frequency (cm^-1)   IR Intensity (km/mol)
    123.45               0.123
    ...
    
    Args:
        ir_file: Path to IR.dat file
        result: Dictionary to update with parsed data
        
    Returns:
        Updated result dictionary
    """
    try:
        with open(ir_file, 'r') as f:
            lines = f.readlines()
        
        frequencies = []
        intensities = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('--'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                    # Only include positive frequencies
                    if freq > 0:
                        frequencies.append(freq)
                        intensities.append(intensity)
                except ValueError:
                    continue
        
        result['ir_frequencies'] = frequencies
        result['ir_intensities'] = intensities
        result['n_ir_modes'] = len(frequencies)
        
        return result
        
    except Exception as e:
        logging.warning(f"Failed to parse IR.dat: {e}")
        return result


def _parse_raman_from_outcar(outcar_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse phonon frequency data directly from OUTCAR.
    
    VASP DFPT outputs phonon frequencies in OUTCAR in the format:
    "   1 f  =    X.XXXXX THz    X.XXXXX 2PiTHz   X.XXXXX cm-1    X.XXXXX meV"
    
    Note: OUTCAR may contain frequencies listed multiple times (summary sections).
    This function extracts the first complete set of phonon frequencies.
    
    Args:
        outcar_path: Path to OUTCAR file
        result: Dictionary to update with parsed data
        
    Returns:
        Updated result dictionary
    """
    import re
    
    try:
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        
        # Look for phonon frequency data in OUTCAR
        # Pattern: "   N f  =    X.XXXXX THz    X.XXXXX 2PiTHz   X.XXXXX cm-1    X.XXXXX meV"
        # or for imaginary: "   N f/i=    X.XXXXX THz ..."
        
        # Pattern for real frequencies with mode number
        freq_pattern = re.compile(
            r'^\s*(\d+)\s+f\s*=\s*([\d.]+)\s+THz\s+([\d.]+)\s+2PiTHz\s+([\d.]+)\s+cm-1\s+([\d.]+)\s+meV'
        )
        # Pattern for imaginary frequencies (acoustic modes at Gamma)
        imag_pattern = re.compile(
            r'^\s*(\d+)\s+f/i\s*=\s*([\d.]+)\s+THz'
        )
        
        frequencies = {}  # Use dict to track mode_number -> frequency
        imaginary_modes = set()
        
        for line in lines:
            # Check for real frequency
            match = freq_pattern.match(line)
            if match:
                mode_num = int(match.group(1))
                freq_cm = float(match.group(4))  # cm-1 value
                
                # Only store the first occurrence of each mode (avoid duplicates)
                if mode_num not in frequencies:
                    frequencies[mode_num] = freq_cm
                continue
            
            # Check for imaginary frequency (acoustic modes)
            imag_match = imag_pattern.match(line)
            if imag_match:
                mode_num = int(imag_match.group(1))
                imaginary_modes.add(mode_num)
        
        # Sort by mode number and filter
        sorted_modes = sorted(frequencies.items(), key=lambda x: x[0])
        
        # Filter out very low frequency modes (< 5 cm-1, likely acoustic)
        # and imaginary modes
        optical_frequencies = []
        for mode_num, freq in sorted_modes:
            if mode_num not in imaginary_modes and freq > 5.0:
                optical_frequencies.append(freq)
        
        if optical_frequencies:
            result['raman_frequencies'] = optical_frequencies
            result['n_raman_modes'] = len(optical_frequencies)
            # Note: Raman activities require additional calculation
            # (second-order response or symmetry analysis)
            result['raman_activities'] = [0.0] * len(optical_frequencies)  # Placeholder
            result['has_raman_data'] = True
            logging.info(f"Parsed {len(optical_frequencies)} optical phonon modes from OUTCAR "
                        f"(filtered from {len(frequencies)} total modes, "
                        f"{len(imaginary_modes)} imaginary/acoustic)")
        else:
            logging.info(f"Found {len(frequencies)} modes but no optical modes above threshold")
        
        return result
        
    except Exception as e:
        logging.warning(f"Failed to parse OUTCAR for phonon data: {e}")
        return result


def parse_raman_from_vaspkit(calc_dir: str, material_id: str) -> Dict[str, Any]:
    """
    Run vaspkit to analyze Raman/IR spectra and parse results.
    
    This is the main entry point for Raman analysis, similar to parse_bandgap_from_vaspkit.
    
    Args:
        calc_dir: Directory containing VASP calculation outputs
        material_id: Material identifier
        
    Returns:
        Dictionary with id, raman_frequencies, raman_activities, etc.
    """
    result = extract_raman_spectra(calc_dir, material_id)
    result['id'] = material_id
    return result
