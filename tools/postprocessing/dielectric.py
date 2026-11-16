"""
Dielectric function and optical properties extraction module.

Provides functions to extract:
- Static dielectric tensor (LEPSILON)
- Frequency-dependent dielectric function (LOPTICS)
- Optical properties (absorption, reflectivity, refractive index)
"""
import os
import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class DielectricResult:
    """Container for dielectric function results."""
    # Static dielectric tensor (3x3)
    epsilon_static: Optional[np.ndarray] = None
    epsilon_ionic: Optional[np.ndarray] = None  # Ionic contribution
    epsilon_electronic: Optional[np.ndarray] = None  # Electronic contribution
    
    # Frequency-dependent dielectric function
    energy: Optional[np.ndarray] = None  # Energy grid (eV)
    epsilon_real: Optional[np.ndarray] = None  # Real part (3x3 for each energy)
    epsilon_imag: Optional[np.ndarray] = None  # Imaginary part
    
    # Derived optical properties
    absorption: Optional[np.ndarray] = None  # Absorption coefficient
    reflectivity: Optional[np.ndarray] = None
    refractive_index: Optional[np.ndarray] = None
    
    # Born effective charges (LEPSILON)
    born_charges: Optional[Dict[int, np.ndarray]] = None
    
    # Metadata
    is_frequency_dependent: bool = False
    n_frequencies: int = 0


def extract_dielectric_tensor(calc_dir: str) -> Optional[DielectricResult]:
    """
    Extract dielectric tensor from OUTCAR.
    
    Works for both LEPSILON (static) and LOPTICS (frequency-dependent) calculations.
    
    Args:
        calc_dir: Directory containing OUTCAR
        
    Returns:
        DielectricResult object or None
    """
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar_path):
        logger.warning(f"OUTCAR not found in {calc_dir}")
        return None
    
    result = DielectricResult()
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Extract static dielectric tensor (from LEPSILON)
        result.epsilon_static = _extract_static_dielectric(content)
        
        # Extract ionic contribution (DFPT)
        result.epsilon_ionic = _extract_ionic_dielectric(content)
        
        # Extract electronic contribution
        result.epsilon_electronic = _extract_electronic_dielectric(content)
        
        # Extract Born effective charges
        result.born_charges = _extract_born_charges(content)
        
        if result.epsilon_static is not None:
            logger.info(f"Extracted static dielectric tensor from OUTCAR")
            logger.info(f"  ε_xx = {result.epsilon_static[0,0]:.4f}, "
                       f"ε_yy = {result.epsilon_static[1,1]:.4f}, "
                       f"ε_zz = {result.epsilon_static[2,2]:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract dielectric tensor: {e}")
        return None


def _extract_static_dielectric(content: str) -> Optional[np.ndarray]:
    """Extract static dielectric tensor from OUTCAR content."""
    # Pattern for "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)"
    patterns = [
        r"MACROSCOPIC STATIC DIELECTRIC TENSOR.*?(?:including local field effects)?.*?\n.*?\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        r"HEAD OF MICROSCOPIC STATIC DIELECTRIC TENSOR.*?\n.*?\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            values = [float(match.group(i)) for i in range(1, 10)]
            return np.array(values).reshape(3, 3)
    
    return None


def _extract_ionic_dielectric(content: str) -> Optional[np.ndarray]:
    """Extract ionic contribution to dielectric tensor."""
    pattern = r"MACROSCOPIC STATIC DIELECTRIC TENSOR.*?IONIC CONTRIBUTION.*?\n.*?\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        values = [float(match.group(i)) for i in range(1, 10)]
        return np.array(values).reshape(3, 3)
    return None


def _extract_electronic_dielectric(content: str) -> Optional[np.ndarray]:
    """Extract electronic contribution to dielectric tensor."""
    # This is typically the first occurrence before ionic contribution
    pattern = r"MACROSCOPIC STATIC DIELECTRIC TENSOR \(including local field effects in DFT\)\s*\n\s*-+\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        values = [float(match.group(i)) for i in range(1, 10)]
        return np.array(values).reshape(3, 3)
    return None


def _extract_born_charges(content: str) -> Optional[Dict[int, np.ndarray]]:
    """Extract Born effective charges from OUTCAR."""
    born_charges = {}
    
    # Pattern for Born effective charge tensor for each ion
    pattern = r"BORN EFFECTIVE CHARGES.*?ion\s+(\d+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    
    for match in re.finditer(pattern, content, re.DOTALL):
        ion_idx = int(match.group(1)) - 1  # Convert to 0-indexed
        values = [float(match.group(i)) for i in range(2, 11)]
        born_charges[ion_idx] = np.array(values).reshape(3, 3)
    
    return born_charges if born_charges else None


def extract_optical_properties(calc_dir: str) -> Optional[DielectricResult]:
    """
    Extract frequency-dependent optical properties from vasprun.xml.
    
    For LOPTICS calculations.
    
    Args:
        calc_dir: Directory containing vasprun.xml
        
    Returns:
        DielectricResult with optical properties
    """
    vasprun_path = os.path.join(calc_dir, 'vasprun.xml')
    if not os.path.exists(vasprun_path):
        logger.warning(f"vasprun.xml not found in {calc_dir}")
        return None
    
    try:
        tree = ET.parse(vasprun_path)
        root = tree.getroot()
        
        result = DielectricResult()
        result.is_frequency_dependent = True
        
        # Find dielectric function data
        for dielectric in root.iter('dielectricfunction'):
            imag_data = dielectric.find(".//imag/array/set")
            real_data = dielectric.find(".//real/array/set")
            
            if imag_data is not None and real_data is not None:
                # Parse energy and dielectric components
                energies = []
                eps_real = []
                eps_imag = []
                
                for r_row, i_row in zip(real_data.findall('r'), imag_data.findall('r')):
                    r_vals = [float(x) for x in r_row.text.split()]
                    i_vals = [float(x) for x in i_row.text.split()]
                    
                    energies.append(r_vals[0])
                    # Columns: energy, xx, yy, zz, xy, yz, zx
                    eps_real.append(r_vals[1:7])
                    eps_imag.append(i_vals[1:7])
                
                result.energy = np.array(energies)
                result.epsilon_real = np.array(eps_real)
                result.epsilon_imag = np.array(eps_imag)
                result.n_frequencies = len(energies)
                
                # Calculate optical properties
                result = _calculate_optical_properties(result)
                
                logger.info(f"Extracted optical properties with {result.n_frequencies} frequency points")
                break
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract optical properties: {e}")
        return None


def _calculate_optical_properties(result: DielectricResult) -> DielectricResult:
    """Calculate derived optical properties from dielectric function."""
    if result.epsilon_real is None or result.epsilon_imag is None:
        return result
    
    # Use xx component for simplicity (could extend to full tensor)
    eps1 = result.epsilon_real[:, 0]  # Real part (xx)
    eps2 = result.epsilon_imag[:, 0]  # Imaginary part (xx)
    
    # Refractive index: n = sqrt((sqrt(eps1^2 + eps2^2) + eps1) / 2)
    n = np.sqrt((np.sqrt(eps1**2 + eps2**2) + eps1) / 2)
    
    # Extinction coefficient: k = sqrt((sqrt(eps1^2 + eps2^2) - eps1) / 2)
    k = np.sqrt((np.sqrt(eps1**2 + eps2**2) - eps1) / 2)
    
    # Absorption coefficient: alpha = 2 * omega * k / c (in cm^-1)
    # omega = E / hbar, with E in eV
    hbar_c = 1973.27  # eV*Angstrom
    alpha = 2 * result.energy * k / hbar_c * 1e8  # Convert to cm^-1
    
    # Reflectivity: R = ((n-1)^2 + k^2) / ((n+1)^2 + k^2)
    reflectivity = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)
    
    result.refractive_index = n
    result.absorption = alpha
    result.reflectivity = reflectivity
    
    return result


def extract_born_effective_charges(calc_dir: str) -> Optional[Dict[int, np.ndarray]]:
    """
    Extract Born effective charges from OUTCAR.
    
    Born effective charges are obtained from LEPSILON/DFPT calculations.
    They are important for understanding ionic contributions to dielectric response.
    
    Args:
        calc_dir: Directory containing OUTCAR
        
    Returns:
        Dictionary mapping atom index (0-based) to 3x3 charge tensor
    """
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar_path):
        logger.warning(f"OUTCAR not found in {calc_dir}")
        return None
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        return _extract_born_charges(content)
        
    except Exception as e:
        logger.error(f"Failed to extract Born effective charges: {e}")
        return None


def get_average_dielectric_constant(result: DielectricResult) -> Dict[str, float]:
    """
    Calculate average dielectric constants from dielectric result.
    
    For isotropic materials, returns the trace average.
    For anisotropic materials, returns individual components.
    
    Args:
        result: DielectricResult object
        
    Returns:
        Dictionary with dielectric constant values
    """
    output = {}
    
    if result.epsilon_static is not None:
        eps = result.epsilon_static
        output['epsilon_static_xx'] = eps[0, 0]
        output['epsilon_static_yy'] = eps[1, 1]
        output['epsilon_static_zz'] = eps[2, 2]
        output['epsilon_static_avg'] = np.trace(eps) / 3
        
        # Check isotropy
        diag = np.diag(eps)
        if np.std(diag) / np.mean(diag) < 0.05:
            output['is_isotropic'] = True
        else:
            output['is_isotropic'] = False
    
    if result.epsilon_electronic is not None:
        eps_e = result.epsilon_electronic
        output['epsilon_infinity_xx'] = eps_e[0, 0]
        output['epsilon_infinity_yy'] = eps_e[1, 1]
        output['epsilon_infinity_zz'] = eps_e[2, 2]
        output['epsilon_infinity_avg'] = np.trace(eps_e) / 3
    
    if result.epsilon_ionic is not None:
        eps_i = result.epsilon_ionic
        output['epsilon_ionic_xx'] = eps_i[0, 0]
        output['epsilon_ionic_yy'] = eps_i[1, 1]
        output['epsilon_ionic_zz'] = eps_i[2, 2]
        output['epsilon_ionic_avg'] = np.trace(eps_i) / 3
    
    return output


def extract_frequency_dependent_dielectric(calc_dir: str) -> Optional[DielectricResult]:
    """
    Extract frequency-dependent dielectric function from LOPTICS calculation.
    
    This is an alias for extract_optical_properties for clearer API.
    
    Args:
        calc_dir: Directory containing vasprun.xml from LOPTICS calculation
        
    Returns:
        DielectricResult with frequency-dependent data
    """
    return extract_optical_properties(calc_dir)
