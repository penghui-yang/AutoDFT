"""
GW quasiparticle calculation analysis module.

Provides functions to:
- Extract quasiparticle (QP) energies from GW calculations
- Calculate GW bandgap
- Analyze QP corrections to DFT eigenvalues
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
class QPEnergy:
    """Quasiparticle energy at a k-point."""
    kpoint_index: int
    kpoint_coords: Tuple[float, float, float]
    band_index: int
    dft_energy: float  # eV (DFT eigenvalue)
    qp_energy: float   # eV (QP energy)
    qp_correction: float  # eV (QP - DFT)
    z_factor: float = 1.0  # Renormalization factor


@dataclass
class GWResult:
    """Container for GW calculation results."""
    # Quasiparticle energies
    qp_energies: List[QPEnergy] = field(default_factory=list)
    
    # Bandgap information
    dft_bandgap: float = 0.0
    gw_bandgap: float = 0.0
    bandgap_correction: float = 0.0
    
    # VBM/CBM information
    dft_vbm: float = 0.0
    dft_cbm: float = 0.0
    gw_vbm: float = 0.0
    gw_cbm: float = 0.0
    
    # Gap type
    is_direct_gap: bool = False
    vbm_kpoint: Optional[Tuple[float, float, float]] = None
    cbm_kpoint: Optional[Tuple[float, float, float]] = None
    
    # Calculation info
    algorithm: str = ""  # GW0, GW, scGW, EVGW
    n_bands: int = 0
    n_kpoints: int = 0
    
    # Convergence info
    is_converged: bool = False


def extract_gw_results(calc_dir: str) -> Optional[GWResult]:
    """
    Extract GW results from OUTCAR and/or vasprun.xml.
    
    Args:
        calc_dir: Directory containing GW calculation output
        
    Returns:
        GWResult object or None
    """
    result = GWResult()
    
    # Try extracting from OUTCAR first (more detailed)
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if os.path.exists(outcar_path):
        outcar_result = _extract_gw_from_outcar(outcar_path)
        if outcar_result:
            result = outcar_result
    
    # Try vasprun.xml for additional info
    vasprun_path = os.path.join(calc_dir, 'vasprun.xml')
    if os.path.exists(vasprun_path):
        vasprun_result = _extract_gw_from_vasprun(vasprun_path)
        if vasprun_result:
            # Merge vasprun data if OUTCAR extraction incomplete
            if not result.qp_energies and vasprun_result.qp_energies:
                result.qp_energies = vasprun_result.qp_energies
    
    # Calculate bandgap if we have QP energies
    if result.qp_energies:
        result = _calculate_gw_bandgap(result)
    
    if result.gw_bandgap > 0:
        logger.info(f"GW bandgap: {result.gw_bandgap:.3f} eV "
                   f"(DFT: {result.dft_bandgap:.3f} eV, "
                   f"correction: {result.bandgap_correction:+.3f} eV)")
    
    return result


def _extract_gw_from_outcar(outcar_path: str) -> Optional[GWResult]:
    """Extract GW results from OUTCAR."""
    result = GWResult()
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Check for GW algorithm
        if "ALGO = GW0" in content or "ALGO=GW0" in content:
            result.algorithm = "GW0"
        elif "ALGO = GW" in content or "ALGO=GW" in content:
            result.algorithm = "GW"
        elif "ALGO = scGW" in content or "ALGO=scGW" in content:
            result.algorithm = "scGW"
        elif "ALGO = EVGW" in content or "ALGO=EVGW" in content:
            result.algorithm = "EVGW"
        else:
            # Not a GW calculation
            return None
        
        # Extract QP energies from the QP eigenvalues table
        # Pattern: "QP-energies"
        qp_section = _find_qp_section(content)
        if qp_section:
            result.qp_energies = _parse_qp_energies(qp_section)
        
        # Extract number of bands and k-points
        nbands_match = re.search(r"NBANDS\s*=\s*(\d+)", content)
        if nbands_match:
            result.n_bands = int(nbands_match.group(1))
        
        nkpts_match = re.search(r"NKPTS\s*=\s*(\d+)", content)
        if nkpts_match:
            result.n_kpoints = int(nkpts_match.group(1))
        
        # Check convergence
        result.is_converged = "reached required accuracy" in content
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract GW from OUTCAR: {e}")
        return None


def _find_qp_section(content: str) -> Optional[str]:
    """Find the QP eigenvalues section in OUTCAR content."""
    # Look for various GW output formats
    patterns = [
        r"QP-energies.*?(?=\n\n|\Z)",
        r"quasiparticle energies.*?(?=\n\n|\Z)",
        r"QUASIPARTICLE BAND STRUCTURE.*?(?=\n\n|\Z)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            # Get a larger context
            start = match.start()
            end = min(start + 10000, len(content))  # Read up to 10k chars
            return content[start:end]
    
    return None


def _parse_qp_energies(section: str) -> List[QPEnergy]:
    """Parse QP energies from the QP section."""
    qp_energies = []
    
    # Try to parse the standard VASP GW output format
    # Format varies by VASP version, try multiple patterns
    
    # Pattern 1: "band   DFT    QP    Z"
    lines = section.split('\n')
    current_kpoint = 0
    current_kcoords = (0.0, 0.0, 0.0)
    
    for line in lines:
        # Check for k-point header
        kpoint_match = re.match(r'\s*k-point\s+(\d+)\s*:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', line)
        if kpoint_match:
            current_kpoint = int(kpoint_match.group(1))
            current_kcoords = (
                float(kpoint_match.group(2)),
                float(kpoint_match.group(3)),
                float(kpoint_match.group(4))
            )
            continue
        
        # Parse band line: "band  DFT_energy  QP_energy  [Z_factor]"
        band_match = re.match(r'\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)(?:\s+([-\d.]+))?', line)
        if band_match:
            band_idx = int(band_match.group(1))
            dft_e = float(band_match.group(2))
            qp_e = float(band_match.group(3))
            z_factor = float(band_match.group(4)) if band_match.group(4) else 1.0
            
            qp = QPEnergy(
                kpoint_index=current_kpoint,
                kpoint_coords=current_kcoords,
                band_index=band_idx,
                dft_energy=dft_e,
                qp_energy=qp_e,
                qp_correction=qp_e - dft_e,
                z_factor=z_factor
            )
            qp_energies.append(qp)
    
    return qp_energies


def _extract_gw_from_vasprun(vasprun_path: str) -> Optional[GWResult]:
    """Extract GW results from vasprun.xml."""
    result = GWResult()
    
    try:
        tree = ET.parse(vasprun_path)
        root = tree.getroot()
        
        # Check if this is a GW calculation
        for param in root.iter('i'):
            if param.get('name') == 'ALGO':
                algo = param.text.strip() if param.text else ""
                if 'GW' in algo.upper():
                    result.algorithm = algo
                    break
        
        if not result.algorithm:
            return None
        
        # Extract eigenvalues (both DFT and QP if available)
        # GW results are typically in <eigenvalues> or <projected> sections
        
        # This is a simplified extraction - VASP GW output format varies
        eigenvalues_set = root.find('.//eigenvalues/array/set')
        if eigenvalues_set is not None:
            # Parse eigenvalues
            for spin_set in eigenvalues_set.findall('set'):
                for kpt_set in spin_set.findall('set'):
                    kpt_idx = 0
                    for r in kpt_set.findall('r'):
                        values = [float(x) for x in r.text.split()]
                        if len(values) >= 2:
                            # First value is energy, second is occupation
                            energy = values[0]
                            # Store as QP energy (simplified)
                            # In real GW output, DFT and QP are separate
        
        return result if result.algorithm else None
        
    except Exception as e:
        logger.warning(f"Failed to extract GW from vasprun.xml: {e}")
        return None


def _calculate_gw_bandgap(result: GWResult) -> GWResult:
    """Calculate bandgap from QP energies."""
    if not result.qp_energies:
        return result
    
    # Separate occupied and unoccupied states
    # Use a simple heuristic: states with lower energy are occupied
    
    # Get unique k-points
    kpoints = {}
    for qp in result.qp_energies:
        k = qp.kpoint_index
        if k not in kpoints:
            kpoints[k] = {'dft': [], 'qp': [], 'coords': qp.kpoint_coords}
        kpoints[k]['dft'].append(qp.dft_energy)
        kpoints[k]['qp'].append(qp.qp_energy)
    
    # Find VBM and CBM across all k-points
    # Assuming Fermi level is at 0 (shifted)
    all_dft = []
    all_qp = []
    
    for k, data in kpoints.items():
        all_dft.extend(data['dft'])
        all_qp.extend(data['qp'])
    
    # Find occupied/unoccupied boundary
    # Heuristic: look for largest gap
    sorted_qp = sorted(all_qp)
    max_gap = 0
    gap_idx = 0
    for i in range(len(sorted_qp) - 1):
        gap = sorted_qp[i + 1] - sorted_qp[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i
    
    if max_gap > 0.01:  # Minimum gap threshold
        result.gw_vbm = sorted_qp[gap_idx]
        result.gw_cbm = sorted_qp[gap_idx + 1]
        result.gw_bandgap = result.gw_cbm - result.gw_vbm
        
        # Do same for DFT
        sorted_dft = sorted(all_dft)
        max_gap_dft = 0
        gap_idx_dft = 0
        for i in range(len(sorted_dft) - 1):
            gap = sorted_dft[i + 1] - sorted_dft[i]
            if gap > max_gap_dft:
                max_gap_dft = gap
                gap_idx_dft = i
        
        if max_gap_dft > 0.01:
            result.dft_vbm = sorted_dft[gap_idx_dft]
            result.dft_cbm = sorted_dft[gap_idx_dft + 1]
            result.dft_bandgap = result.dft_cbm - result.dft_vbm
        
        result.bandgap_correction = result.gw_bandgap - result.dft_bandgap
    
    return result


def extract_qp_corrections(calc_dir: str) -> Optional[Dict[str, Any]]:
    """
    Extract QP correction statistics from GW calculation.
    
    Returns summary of how much GW shifts eigenvalues relative to DFT.
    
    Args:
        calc_dir: Directory containing GW output
        
    Returns:
        Dictionary with correction statistics
    """
    result = extract_gw_results(calc_dir)
    if not result or not result.qp_energies:
        return None
    
    corrections = [qp.qp_correction for qp in result.qp_energies]
    
    # Separate occupied and unoccupied corrections (heuristic)
    occupied_corr = [qp.qp_correction for qp in result.qp_energies 
                     if qp.dft_energy < 0]  # Below assumed Fermi level
    unoccupied_corr = [qp.qp_correction for qp in result.qp_energies 
                       if qp.dft_energy >= 0]
    
    stats = {
        'mean_correction': np.mean(corrections) if corrections else 0,
        'std_correction': np.std(corrections) if corrections else 0,
        'max_correction': max(corrections) if corrections else 0,
        'min_correction': min(corrections) if corrections else 0,
        'mean_occupied_correction': np.mean(occupied_corr) if occupied_corr else 0,
        'mean_unoccupied_correction': np.mean(unoccupied_corr) if unoccupied_corr else 0,
        'bandgap_correction': result.bandgap_correction,
        'dft_bandgap': result.dft_bandgap,
        'gw_bandgap': result.gw_bandgap,
    }
    
    return stats


def get_gw_incar_settings(
    n_bands: int,
    algo: str = "GW0",
    nomega: int = 50,
    encutgw: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get recommended INCAR settings for GW calculation.
    
    Args:
        n_bands: Number of bands (should be ~4x occupied)
        algo: GW algorithm (GW0, GW, scGW, EVGW)
        nomega: Number of frequency grid points
        encutgw: GW cutoff energy (if None, uses 2/3 of ENCUT)
    
    Returns:
        Dictionary of INCAR parameters
    """
    incar = {
        "ALGO": algo,
        "NBANDS": n_bands,
        "NOMEGA": nomega,
        "NELM": 1 if algo == "GW0" else 8,  # Single iteration for G0W0
        "LOPTICS": ".TRUE.",  # May need for WAVEDER
        "LWAVE": ".FALSE.",  # Don't write WAVECAR from GW
        "PRECFOCK": "Fast",
    }
    
    if encutgw:
        incar["ENCUTGW"] = encutgw
    
    return incar


def get_gw_dft_incar_settings(n_bands: int) -> Dict[str, Any]:
    """
    Get INCAR settings for DFT pre-calculation before GW.
    
    Args:
        n_bands: Number of bands for GW (should be used consistently)
    
    Returns:
        Dictionary of INCAR parameters
    """
    incar = {
        "ALGO": "Normal",
        "PREC": "Accurate",
        "EDIFF": 1e-8,
        "NBANDS": n_bands,
        "LWAVE": ".TRUE.",   # Need WAVECAR for GW
        "LCHARG": ".TRUE.",  # Need CHGCAR
        "LOPTICS": ".TRUE.", # Generate WAVEDER for GW
    }
    
    return incar
