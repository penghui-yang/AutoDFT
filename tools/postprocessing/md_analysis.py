"""
Molecular Dynamics trajectory analysis module.

Provides functions to:
- Parse XDATCAR trajectories
- Extract temperature and energy evolution
- Calculate statistical quantities (RDF, MSD, diffusion coefficients)
"""
import os
import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MDTrajectory:
    """Container for MD trajectory data from XDATCAR."""
    # System information
    system_name: str = ""
    n_atoms: int = 0
    n_frames: int = 0
    elements: List[str] = field(default_factory=list)
    atom_counts: List[int] = field(default_factory=list)
    
    # Lattice vectors (3x3, may be constant or vary with frame)
    lattice: Optional[np.ndarray] = None  # Shape: (3, 3) or (n_frames, 3, 3)
    is_variable_cell: bool = False
    
    # Atomic positions for all frames
    # Shape: (n_frames, n_atoms, 3) in fractional coordinates
    positions: Optional[np.ndarray] = None
    
    # Time information (if available from OUTCAR)
    timestep: float = 1.0  # fs
    times: Optional[np.ndarray] = None  # Time of each frame in fs


@dataclass
class MDStatistics:
    """Container for MD statistical analysis results."""
    # Temperature evolution
    temperatures: Optional[np.ndarray] = None  # K
    mean_temperature: float = 0.0
    std_temperature: float = 0.0
    
    # Energy evolution
    total_energies: Optional[np.ndarray] = None  # eV
    kinetic_energies: Optional[np.ndarray] = None  # eV
    potential_energies: Optional[np.ndarray] = None  # eV
    mean_total_energy: float = 0.0
    std_total_energy: float = 0.0
    energy_drift: float = 0.0  # eV/ps (for NVE check)
    
    # Time array
    times: Optional[np.ndarray] = None  # fs
    
    # Equilibration info
    equilibration_frame: int = 0  # Frame where equilibration is reached
    is_equilibrated: bool = False
    
    # MSD and diffusion (if calculated)
    msd: Optional[np.ndarray] = None  # Mean square displacement
    diffusion_coefficient: Optional[float] = None  # cm^2/s


def parse_xdatcar(calc_dir: str) -> Optional[MDTrajectory]:
    """
    Parse XDATCAR file to extract MD trajectory.
    
    Args:
        calc_dir: Directory containing XDATCAR
        
    Returns:
        MDTrajectory object or None
    """
    xdatcar_path = os.path.join(calc_dir, 'XDATCAR')
    if not os.path.exists(xdatcar_path):
        logger.warning(f"XDATCAR not found in {calc_dir}")
        return None
    
    try:
        with open(xdatcar_path, 'r') as f:
            lines = f.readlines()
        
        trajectory = MDTrajectory()
        
        # Parse header
        trajectory.system_name = lines[0].strip()
        scale = float(lines[1].strip())
        
        # Parse lattice vectors (lines 2-4)
        lattice = np.zeros((3, 3))
        for i in range(3):
            lattice[i] = [float(x) for x in lines[2 + i].split()]
        lattice *= scale
        trajectory.lattice = lattice
        
        # Parse elements and counts (lines 5-6)
        trajectory.elements = lines[5].split()
        trajectory.atom_counts = [int(x) for x in lines[6].split()]
        trajectory.n_atoms = sum(trajectory.atom_counts)
        
        # Parse frames
        positions_list = []
        line_idx = 7
        
        while line_idx < len(lines):
            # Skip "Direct configuration= X" line
            if lines[line_idx].strip().startswith('Direct') or 'configuration' in lines[line_idx].lower():
                line_idx += 1
                continue
            
            # Read positions for one frame
            frame_positions = []
            for i in range(trajectory.n_atoms):
                if line_idx + i >= len(lines):
                    break
                pos = [float(x) for x in lines[line_idx + i].split()[:3]]
                frame_positions.append(pos)
            
            if len(frame_positions) == trajectory.n_atoms:
                positions_list.append(frame_positions)
            
            line_idx += trajectory.n_atoms
        
        trajectory.positions = np.array(positions_list)
        trajectory.n_frames = len(positions_list)
        
        logger.info(f"Parsed XDATCAR: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
        return trajectory
        
    except Exception as e:
        logger.error(f"Failed to parse XDATCAR: {e}")
        return None


def extract_md_statistics(
    calc_dir: str,
    skip_frames: int = 0
) -> Optional[MDStatistics]:
    """
    Extract MD statistics from OUTCAR or OSZICAR.
    
    Args:
        calc_dir: Directory containing OUTCAR/OSZICAR
        skip_frames: Number of initial frames to skip (equilibration)
        
    Returns:
        MDStatistics object or None
    """
    stats = MDStatistics()
    
    # Try OSZICAR first (faster to parse)
    oszicar_path = os.path.join(calc_dir, 'OSZICAR')
    if os.path.exists(oszicar_path):
        _parse_oszicar_for_md(oszicar_path, stats)
    
    # Get more details from OUTCAR
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if os.path.exists(outcar_path):
        _parse_outcar_for_md(outcar_path, stats)
    
    # Calculate statistics (excluding equilibration)
    if stats.temperatures is not None and len(stats.temperatures) > skip_frames:
        temps = stats.temperatures[skip_frames:]
        stats.mean_temperature = float(np.mean(temps))
        stats.std_temperature = float(np.std(temps))
    
    if stats.total_energies is not None and len(stats.total_energies) > skip_frames:
        energies = stats.total_energies[skip_frames:]
        stats.mean_total_energy = float(np.mean(energies))
        stats.std_total_energy = float(np.std(energies))
        
        # Calculate energy drift (for NVE simulations)
        if len(energies) > 10:
            # Linear fit to check drift
            x = np.arange(len(energies))
            slope, _ = np.polyfit(x, energies, 1)
            # Convert to eV/ps assuming typical timestep
            timestep_fs = 1.0  # Will be updated if found in OUTCAR
            stats.energy_drift = slope / timestep_fs * 1000  # eV/ps
    
    # Determine equilibration
    stats.equilibration_frame = skip_frames
    stats.is_equilibrated = skip_frames > 0
    
    return stats


def _parse_oszicar_for_md(oszicar_path: str, stats: MDStatistics):
    """Parse OSZICAR for MD energies and temperatures."""
    temperatures = []
    total_energies = []
    kinetic_energies = []
    
    try:
        with open(oszicar_path, 'r') as f:
            for line in f:
                # MD step line format: "   N       F=  xxx.xxx E0=  xxx.xxx d E =xxx.xxx EK=xxx.xxx SP=xxx.xxx SK=xxx.xxx"
                # Or: "   N T=   xxx.xx E= xxx.xxx F= xxx.xxx E0= xxx.xxx EK= xxx.xxx SP= xxx.xxx SK= xxx.xxx"
                if 'T=' in line or 'EK=' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'T=' and i + 1 < len(parts):
                            try:
                                temperatures.append(float(parts[i + 1]))
                            except ValueError:
                                pass
                        elif part == 'E=' and i + 1 < len(parts):
                            try:
                                total_energies.append(float(parts[i + 1]))
                            except ValueError:
                                pass
                        elif part == 'EK=' and i + 1 < len(parts):
                            try:
                                kinetic_energies.append(float(parts[i + 1]))
                            except ValueError:
                                pass
        
        if temperatures:
            stats.temperatures = np.array(temperatures)
        if total_energies:
            stats.total_energies = np.array(total_energies)
        if kinetic_energies:
            stats.kinetic_energies = np.array(kinetic_energies)
            
    except Exception as e:
        logger.warning(f"Failed to parse OSZICAR for MD: {e}")


def _parse_outcar_for_md(outcar_path: str, stats: MDStatistics):
    """Parse OUTCAR for detailed MD information."""
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Extract POTIM (timestep)
        potim_match = re.search(r'POTIM\s*=\s*([\d.]+)', content)
        if potim_match:
            timestep = float(potim_match.group(1))
            if stats.temperatures is not None:
                n_steps = len(stats.temperatures)
                stats.times = np.arange(n_steps) * timestep
        
        # Extract temperatures from OUTCAR if not in OSZICAR
        if stats.temperatures is None or len(stats.temperatures) == 0:
            temp_pattern = re.compile(r'kin\. lattice\s+EKIN_LAT=\s*[\d.]+\s+\(temperature\s+([\d.]+)\s+K\)')
            temps = [float(m.group(1)) for m in temp_pattern.finditer(content)]
            if temps:
                stats.temperatures = np.array(temps)
        
    except Exception as e:
        logger.warning(f"Failed to parse OUTCAR for MD: {e}")


def calculate_msd(
    trajectory: MDTrajectory,
    atom_indices: Optional[List[int]] = None,
    max_frames: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean square displacement (MSD) from trajectory.
    
    Args:
        trajectory: MDTrajectory object
        atom_indices: Atom indices to include (None for all)
        max_frames: Maximum number of frame intervals to calculate
        
    Returns:
        Tuple of (time_lags, msd_values)
    """
    if trajectory.positions is None:
        logger.warning("No position data in trajectory")
        return np.array([]), np.array([])
    
    positions = trajectory.positions
    n_frames = trajectory.n_frames
    
    if atom_indices is not None:
        positions = positions[:, atom_indices, :]
    
    # Convert to Cartesian coordinates
    if trajectory.lattice is not None:
        # Handle both constant and variable cell
        if trajectory.lattice.ndim == 2:
            # Constant cell
            cart_positions = np.einsum('ijk,kl->ijl', positions, trajectory.lattice)
        else:
            # Variable cell - apply each frame's lattice
            cart_positions = np.einsum('ijk,ikl->ijl', positions, trajectory.lattice)
    else:
        cart_positions = positions
    
    # Calculate MSD
    if max_frames is None:
        max_frames = n_frames // 2
    max_frames = min(max_frames, n_frames - 1)
    
    msd = np.zeros(max_frames)
    counts = np.zeros(max_frames)
    
    for dt in range(1, max_frames):
        for t in range(n_frames - dt):
            disp = cart_positions[t + dt] - cart_positions[t]
            # Sum over atoms and components
            msd[dt] += np.sum(disp**2)
            counts[dt] += positions.shape[1]  # Number of atoms
    
    # Average MSD
    msd = np.divide(msd, counts, where=counts > 0)
    
    # Time lags
    time_lags = np.arange(max_frames) * trajectory.timestep
    
    return time_lags, msd


def estimate_diffusion_coefficient(
    time_lags: np.ndarray,
    msd: np.ndarray,
    fit_range: Tuple[float, float] = (0.2, 0.8)
) -> float:
    """
    Estimate diffusion coefficient from MSD using Einstein relation.
    
    D = MSD / (6 * t) for 3D diffusion
    
    Args:
        time_lags: Time lag array (fs)
        msd: Mean square displacement (Å^2)
        fit_range: Fraction of data to use for linear fit (start, end)
        
    Returns:
        Diffusion coefficient in cm^2/s
    """
    n = len(time_lags)
    start_idx = int(n * fit_range[0])
    end_idx = int(n * fit_range[1])
    
    if end_idx <= start_idx + 2:
        logger.warning("Insufficient data for diffusion coefficient fit")
        return 0.0
    
    t = time_lags[start_idx:end_idx]
    m = msd[start_idx:end_idx]
    
    # Linear fit: MSD = 6*D*t
    slope, _ = np.polyfit(t, m, 1)
    
    # Convert from Å^2/fs to cm^2/s
    # 1 Å = 1e-8 cm, 1 fs = 1e-15 s
    # Å^2/fs = 1e-16 cm^2 / 1e-15 s = 0.1 cm^2/s
    D = slope / 6 * 0.1  # cm^2/s
    
    return D
