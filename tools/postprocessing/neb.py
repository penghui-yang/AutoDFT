"""
NEB (Nudged Elastic Band) reaction path analysis module.

Provides functions to:
- Extract NEB energy profile
- Calculate reaction barriers
- Analyze transition state geometry
"""
import os
import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NEBResult:
    """Container for NEB calculation results."""
    # Number of images (excluding endpoints)
    n_images: int = 0
    
    # Energies along the path (eV)
    energies: List[float] = field(default_factory=list)
    # Relative energies (referenced to initial state)
    relative_energies: List[float] = field(default_factory=list)
    
    # Reaction coordinate (normalized 0-1)
    reaction_coordinate: List[float] = field(default_factory=list)
    
    # Tangent forces along path (eV/Å)
    tangent_forces: List[float] = field(default_factory=list)
    
    # Barrier information
    forward_barrier: float = 0.0  # eV (from initial to TS)
    reverse_barrier: float = 0.0  # eV (from final to TS)
    reaction_energy: float = 0.0  # eV (final - initial)
    
    # Transition state info
    ts_image_index: int = -1
    ts_energy: float = 0.0
    ts_is_converged: bool = False
    
    # Convergence info
    is_converged: bool = False
    max_force: float = 0.0
    n_steps: int = 0
    
    # Climbing image info (CI-NEB)
    is_climbing_image: bool = False
    climbing_image_index: int = -1


def extract_neb_path(calc_dir: str) -> Optional[NEBResult]:
    """
    Extract NEB reaction path from multi-image calculation.
    
    Expects directory structure:
    calc_dir/
        00/  (initial state)
        01/
        02/
        ...
        0N/  (final state)
    
    Args:
        calc_dir: Base directory containing image subdirectories
        
    Returns:
        NEBResult object or None
    """
    result = NEBResult()
    
    # Find all image directories
    image_dirs = []
    for item in sorted(os.listdir(calc_dir)):
        if re.match(r'^\d{2}$', item):
            image_path = os.path.join(calc_dir, item)
            if os.path.isdir(image_path):
                image_dirs.append((int(item), image_path))
    
    if len(image_dirs) < 3:
        # Also try VASP 6 style with IMAGES directory
        images_dir = os.path.join(calc_dir, 'IMAGES')
        if os.path.isdir(images_dir):
            return _extract_neb_vasp6(images_dir)
        
        logger.warning(f"Not enough NEB images found in {calc_dir}")
        return None
    
    image_dirs.sort(key=lambda x: x[0])
    result.n_images = len(image_dirs) - 2  # Exclude endpoints
    
    # Extract energy from each image
    for idx, image_dir in image_dirs:
        energy = _extract_image_energy(image_dir)
        if energy is not None:
            result.energies.append(energy)
        else:
            logger.warning(f"Could not extract energy from image {idx}")
            result.energies.append(0.0)
    
    # Calculate relative energies
    if result.energies:
        e_init = result.energies[0]
        result.relative_energies = [e - e_init for e in result.energies]
    
    # Calculate reaction coordinate
    n_total = len(result.energies)
    if n_total > 1:
        result.reaction_coordinate = [i / (n_total - 1) for i in range(n_total)]
    
    # Calculate barriers
    result = calculate_barrier(result)
    
    # Check convergence from OUTCAR in main directory or first moving image
    main_outcar = os.path.join(calc_dir, 'OUTCAR')
    if os.path.exists(main_outcar):
        result.is_converged, result.max_force, result.n_steps = _check_neb_convergence(main_outcar)
    
    logger.info(f"Extracted NEB path with {n_total} images")
    logger.info(f"  Forward barrier: {result.forward_barrier:.3f} eV")
    logger.info(f"  Reverse barrier: {result.reverse_barrier:.3f} eV")
    logger.info(f"  Reaction energy: {result.reaction_energy:.3f} eV")
    
    return result


def _extract_neb_vasp6(images_dir: str) -> Optional[NEBResult]:
    """Extract NEB data from VASP 6 style IMAGES directory."""
    result = NEBResult()
    
    # VASP 6 stores all images in IMAGES/XX directories
    image_dirs = []
    for item in sorted(os.listdir(images_dir)):
        if re.match(r'^\d{2}$', item):
            image_path = os.path.join(images_dir, item)
            if os.path.isdir(image_path):
                image_dirs.append((int(item), image_path))
    
    if len(image_dirs) < 3:
        return None
    
    image_dirs.sort(key=lambda x: x[0])
    result.n_images = len(image_dirs) - 2
    
    for idx, image_dir in image_dirs:
        energy = _extract_image_energy(image_dir)
        result.energies.append(energy if energy is not None else 0.0)
    
    if result.energies:
        e_init = result.energies[0]
        result.relative_energies = [e - e_init for e in result.energies]
    
    n_total = len(result.energies)
    result.reaction_coordinate = [i / (n_total - 1) for i in range(n_total)]
    
    result = calculate_barrier(result)
    return result


def _extract_image_energy(image_dir: str) -> Optional[float]:
    """Extract final energy from an image directory."""
    outcar_path = os.path.join(image_dir, 'OUTCAR')
    
    if not os.path.exists(outcar_path):
        # Try vasprun.xml
        vasprun_path = os.path.join(image_dir, 'vasprun.xml')
        if os.path.exists(vasprun_path):
            return _extract_energy_from_vasprun(vasprun_path)
        return None
    
    try:
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        
        # Find last occurrence of TOTEN
        for line in reversed(lines):
            if "free  energy   TOTEN" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "TOTEN" and i + 2 < len(parts):
                        return float(parts[i + 2])
                break
        
        # Alternative: sigma->0 energy
        for line in reversed(lines):
            if "energy  without entropy" in line:
                parts = line.split()
                return float(parts[-1])
        
    except Exception as e:
        logger.warning(f"Failed to extract energy from {image_dir}: {e}")
    
    return None


def _extract_energy_from_vasprun(vasprun_path: str) -> Optional[float]:
    """Extract energy from vasprun.xml."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(vasprun_path)
        root = tree.getroot()
        
        # Find last calculation energy
        for calculation in root.findall('.//calculation'):
            energy_elem = calculation.find('.//energy/i[@name="e_fr_energy"]')
            if energy_elem is not None:
                return float(energy_elem.text)
        
    except Exception as e:
        logger.warning(f"Failed to parse vasprun.xml: {e}")
    
    return None


def _check_neb_convergence(outcar_path: str) -> Tuple[bool, float, int]:
    """Check NEB convergence from OUTCAR."""
    is_converged = False
    max_force = float('inf')
    n_steps = 0
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Check for convergence message
        if "reached required accuracy" in content:
            is_converged = True
        
        # Extract max force
        force_pattern = re.compile(r'RMS\s+\d+\.\d+E[+-]\d+\s+\d+\.\d+E[+-]\d+\s+(\d+\.\d+E[+-]\d+)')
        forces = force_pattern.findall(content)
        if forces:
            max_force = float(forces[-1])
        
        # Count NEB steps
        n_steps = len(re.findall(r'CHAIN \+', content))
        
    except Exception as e:
        logger.warning(f"Failed to check NEB convergence: {e}")
    
    return is_converged, max_force, n_steps


def calculate_barrier(result: NEBResult) -> NEBResult:
    """
    Calculate reaction barriers from NEB energies.
    
    Args:
        result: NEBResult with energies populated
        
    Returns:
        NEBResult with barrier information added
    """
    if not result.relative_energies:
        return result
    
    energies = result.relative_energies
    
    # Find transition state (maximum energy)
    max_energy = max(energies)
    ts_index = energies.index(max_energy)
    
    result.ts_image_index = ts_index
    result.ts_energy = result.energies[ts_index] if result.energies else 0.0
    
    # Forward barrier: TS - initial
    result.forward_barrier = energies[ts_index] - energies[0]
    
    # Reverse barrier: TS - final
    result.reverse_barrier = energies[ts_index] - energies[-1]
    
    # Reaction energy: final - initial
    result.reaction_energy = energies[-1] - energies[0]
    
    return result


def interpolate_images(
    initial_poscar: str,
    final_poscar: str,
    n_images: int,
    output_dir: str,
    method: str = "linear"
) -> List[str]:
    """
    Generate intermediate NEB images by interpolation.
    
    Args:
        initial_poscar: Path to initial state POSCAR
        final_poscar: Path to final state POSCAR
        n_images: Number of intermediate images
        output_dir: Directory to write image POSCARs
        method: Interpolation method ("linear" or "idpp")
        
    Returns:
        List of paths to generated POSCAR files
    """
    try:
        # Parse initial and final structures
        with open(initial_poscar, 'r') as f:
            initial_lines = f.readlines()
        with open(final_poscar, 'r') as f:
            final_lines = f.readlines()
        
        # Extract positions (assuming same cell and species)
        initial_pos = _extract_positions(initial_lines)
        final_pos = _extract_positions(final_lines)
        
        if initial_pos is None or final_pos is None:
            logger.error("Failed to parse POSCAR files")
            return []
        
        # Linear interpolation
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        for i in range(n_images + 2):
            # Fraction along path (0 = initial, 1 = final)
            fraction = i / (n_images + 1)
            
            # Interpolate positions
            interp_pos = initial_pos + fraction * (final_pos - initial_pos)
            
            # Write POSCAR
            image_dir = os.path.join(output_dir, f"{i:02d}")
            os.makedirs(image_dir, exist_ok=True)
            poscar_path = os.path.join(image_dir, "POSCAR")
            
            _write_poscar(poscar_path, initial_lines, interp_pos)
            output_files.append(poscar_path)
        
        logger.info(f"Generated {n_images + 2} NEB images in {output_dir}")
        return output_files
        
    except Exception as e:
        logger.error(f"Failed to interpolate NEB images: {e}")
        return []


def _extract_positions(lines: List[str]) -> Optional[np.ndarray]:
    """Extract atomic positions from POSCAR lines."""
    try:
        # Find number of atoms
        atom_counts = [int(x) for x in lines[6].split()]
        n_atoms = sum(atom_counts)
        
        # Check for Selective dynamics
        start_line = 7
        if lines[7].strip()[0].upper() == 'S':
            start_line = 8
        
        # Skip coordinate type line
        start_line += 1
        
        positions = []
        for i in range(n_atoms):
            parts = lines[start_line + i].split()
            pos = [float(parts[j]) for j in range(3)]
            positions.append(pos)
        
        return np.array(positions)
        
    except Exception as e:
        logger.warning(f"Failed to extract positions: {e}")
        return None


def _write_poscar(path: str, template_lines: List[str], positions: np.ndarray):
    """Write POSCAR file with new positions."""
    with open(path, 'w') as f:
        # Write header (lines 0-6 or 0-7 if Selective dynamics)
        start_line = 7
        if len(template_lines) > 7 and template_lines[7].strip()[0].upper() == 'S':
            start_line = 8
            for i in range(8):
                f.write(template_lines[i])
        else:
            for i in range(7):
                f.write(template_lines[i])
        
        # Write coordinate type
        f.write(template_lines[start_line])
        start_line += 1
        
        # Write positions
        for pos in positions:
            f.write(f"  {pos[0]:.16f}  {pos[1]:.16f}  {pos[2]:.16f}\n")


def prepare_neb_directories(
    calc_dir: str,
    n_images: int,
    initial_poscar: str,
    final_poscar: str,
    copy_files: List[str] = None
) -> List[str]:
    """
    Prepare NEB directory structure with image POSCARs.
    
    Creates:
    calc_dir/
        00/POSCAR  (initial)
        01/POSCAR  (interpolated)
        ...
        0N/POSCAR  (final)
    
    Args:
        calc_dir: Base directory for NEB calculation
        n_images: Number of intermediate images
        initial_poscar: Path to initial state POSCAR
        final_poscar: Path to final state POSCAR
        copy_files: Additional files to copy to each image dir (e.g., POTCAR)
    
    Returns:
        List of image directory paths
    """
    os.makedirs(calc_dir, exist_ok=True)
    
    # Generate interpolated images
    image_files = interpolate_images(
        initial_poscar=initial_poscar,
        final_poscar=final_poscar,
        n_images=n_images,
        output_dir=calc_dir,
        method="linear"
    )
    
    if not image_files:
        logger.error("Failed to generate NEB images")
        return []
    
    # Get image directories
    image_dirs = [os.path.dirname(f) for f in image_files]
    
    # Copy additional files if specified
    if copy_files:
        import shutil
        for image_dir in image_dirs:
            for file_path in copy_files:
                if os.path.exists(file_path):
                    shutil.copy(file_path, image_dir)
    
    logger.info(f"Prepared {len(image_dirs)} NEB image directories in {calc_dir}")
    return image_dirs


def get_neb_incar_settings(
    n_images: int,
    climbing_image: bool = True,
    spring_constant: float = -5.0,
    ediffg: float = -0.05
) -> Dict[str, Any]:
    """
    Get recommended INCAR settings for NEB calculation.
    
    Args:
        n_images: Number of intermediate images
        climbing_image: Whether to use climbing image method
        spring_constant: Spring constant between images (eV/Å²)
        ediffg: Force convergence criterion (eV/Å)
    
    Returns:
        Dictionary of INCAR parameters
    """
    incar = {
        "IBRION": 3,           # Required for NEB
        "POTIM": 0,            # Let VASP handle step size
        "IMAGES": n_images,    # Number of intermediate images
        "SPRING": spring_constant,
        "LCLIMB": ".TRUE." if climbing_image else ".FALSE.",
        "ICHAIN": 0,           # NEB method
        "EDIFFG": ediffg,
        "NSW": 200,            # Max ionic steps
        "IOPT": 0,             # VASP optimizer (or 1 for VTST LBFGS)
        "LNEBCELL": ".FALSE.", # Fixed cell
        "LREAL": "Auto",       # For efficiency with larger cells
    }
    
    return incar


def plot_neb_path(
    result: NEBResult,
    output_path: Optional[str] = None,
    title: str = "NEB Reaction Path"
) -> Optional[str]:
    """
    Plot NEB energy profile.
    
    Args:
        result: NEBResult with energies
        output_path: Path to save figure (optional)
        title: Plot title
    
    Returns:
        Path to saved figure or None
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot energy profile
        ax.plot(result.reaction_coordinate, result.relative_energies, 
                'o-', linewidth=2, markersize=8, color='blue')
        
        # Mark transition state
        if result.ts_image_index >= 0:
            ts_x = result.reaction_coordinate[result.ts_image_index]
            ts_y = result.relative_energies[result.ts_image_index]
            ax.axhline(y=ts_y, color='red', linestyle='--', alpha=0.5)
            ax.annotate(f'TS: {result.forward_barrier:.3f} eV',
                       xy=(ts_x, ts_y), xytext=(ts_x + 0.1, ts_y + 0.05),
                       fontsize=10, color='red')
        
        # Mark initial and final states
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        if result.relative_energies:
            ax.axhline(y=result.relative_energies[-1], color='gray', 
                      linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Reaction Coordinate', fontsize=12)
        ax.set_ylabel('Relative Energy (eV)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add barrier annotation
        barrier_text = (f"Forward barrier: {result.forward_barrier:.3f} eV\n"
                       f"Reverse barrier: {result.reverse_barrier:.3f} eV\n"
                       f"Reaction energy: {result.reaction_energy:.3f} eV")
        ax.text(0.02, 0.98, barrier_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            logger.info(f"NEB path plot saved to {output_path}")
            plt.close()
            return output_path
        else:
            plt.close()
            return None
            
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None
    except Exception as e:
        logger.error(f"Failed to plot NEB path: {e}")
        return None


def extract_neb_forces(calc_dir: str) -> Optional[List[float]]:
    """
    Extract tangent forces along NEB path.
    
    Args:
        calc_dir: NEB calculation directory
    
    Returns:
        List of tangent forces for each image
    """
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar_path):
        return None
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Pattern for NEB forces
        # Look for "tangent forces" in VASP NEB output
        forces = []
        pattern = r"tangent\s+\d+\s+([-\d.E+]+)"
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        if matches:
            forces = [float(m) for m in matches]
        
        return forces if forces else None
        
    except Exception as e:
        logger.warning(f"Failed to extract NEB forces: {e}")
        return None
