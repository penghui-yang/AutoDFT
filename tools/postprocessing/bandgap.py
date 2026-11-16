"""
Bandgap extraction and analysis module.

Provides functions to extract bandgap information from VASP calculations
using VASPKIT or direct parsing of EIGENVAL/vasprun.xml.
"""
import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


@dataclass
class BandgapResult:
    """Result of bandgap extraction."""
    bandgap: float  # eV
    is_direct: bool
    vbm_energy: float  # eV
    cbm_energy: float  # eV
    vbm_kpoint: Tuple[float, float, float]
    cbm_kpoint: Tuple[float, float, float]
    is_metal: bool = False
    is_semimetal: bool = False
    spin_polarized: bool = False
    spin_up_gap: Optional[float] = None
    spin_down_gap: Optional[float] = None
    fermi_energy: Optional[float] = None  # Fermi energy (eV)
    electronic_type: Optional[str] = None  # "n-type", "p-type", "intrinsic", "metal"


@dataclass
class ElectronicTypeResult:
    """
    Result of electronic type classification.
    
    导电类型分类结果，用于判断材料是 n-type、p-type 还是绝缘体
    """
    electronic_type: str  # "n-type", "p-type", "intrinsic", "metal"
    fermi_energy: float  # Fermi level (eV)
    vbm_energy: float  # Valence band maximum (eV)
    cbm_energy: float  # Conduction band minimum (eV)
    bandgap: float  # Band gap (eV)
    fermi_to_vbm: float  # E_f - VBM (eV), positive means above VBM
    fermi_to_cbm: float  # CBM - E_f (eV), positive means below CBM
    confidence: str  # "high", "medium", "low"
    notes: str = ""  # Additional notes


def parse_bandgap_from_vaspkit(calc_dir: str, material_id: str) -> Dict[str, Any]:
    """
    Run vaspkit to analyze band gap and parse results.
    
    Handles two VASPKIT output formats:
    1. Non-spin-polarized: Has "Band Character: Direct/Indirect" line
    2. Spin-polarized (ISPIN=2): Has "Spin Channel: <UP> <DOWN> <TOTAL>" format
    
    For metallic cases, distinguishes between Metal and Semi-metal based on
    whether VBM and CBM are at the same k-point location.
    
    FALLBACK: If vaspkit fails to generate BAND_GAP file (e.g. for metallic
    systems detected incorrectly with ISPIN=1), falls back to direct EIGENVAL
    parsing using occupation numbers. This handles both ISPIN=1 and ISPIN=2.
    
    Args:
        calc_dir: Directory containing VASP calculation outputs
        material_id: Material identifier
        
    Returns:
        Dictionary with id, calculated_character, and calculated_gap
    """
    from execution.job_manager import run_command
    
    logger.info(f"Analyzing bandgap for {material_id} in {calc_dir}")
    
    # Default result (returned on complete failure)
    default_result = {
        'id': material_id,
        'calculated_character': "ExtractionFailed",
        'calculated_gap': None,
        'extraction_error': "All bandgap extraction methods failed"
    }
    
    # Try vaspkit first
    vaspkit_result = None
    try:
        # vaspkit task 211 (Band-Structure -> Extract Band Data)
        run_command("vaspkit -task 211", cwd=calc_dir, shell=True)
        
        # Parse the BAND_GAP file
        band_gap_file = os.path.join(calc_dir, 'BAND_GAP')
        if os.path.exists(band_gap_file):
            vaspkit_result = _parse_band_gap_file(band_gap_file, material_id)
            # Check if parsing was successful (not just default zeros)
            if (vaspkit_result.get('calculated_gap') is not None and 
                vaspkit_result.get('calculated_character') not in (None, "None")):
                return vaspkit_result
            else:
                logger.warning(f"BAND_GAP file parsing incomplete for {material_id}, trying EIGENVAL fallback")
        else:
            logger.warning(f"BAND_GAP file not found for {material_id}, trying EIGENVAL fallback")
    except Exception as e:
        logger.warning(f"vaspkit failed for {material_id}: {e}, trying EIGENVAL fallback")
    
    # FALLBACK: Direct EIGENVAL parsing
    eigenval_path = os.path.join(calc_dir, 'EIGENVAL')
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    
    if os.path.exists(eigenval_path):
        try:
            # Get Fermi energy from OUTCAR for reference
            efermi = _extract_fermi_from_outcar(outcar_path) if os.path.exists(outcar_path) else None
            
            eigenval_result = extract_bandgap_from_eigenval(eigenval_path, efermi)
            if eigenval_result is not None:
                # Convert BandgapResult to dict format
                if eigenval_result.is_metal:
                    character = "Metal"
                elif eigenval_result.is_direct:
                    character = "Direct"
                else:
                    character = "Indirect"
                
                logger.info(f"EIGENVAL fallback for {material_id}: {character}, {eigenval_result.bandgap:.4f} eV")
                return {
                    'id': material_id,
                    'calculated_character': character,
                    'calculated_gap': eigenval_result.bandgap
                }
        except Exception as e:
            logger.error(f"EIGENVAL fallback also failed for {material_id}: {e}")
    
    # If vaspkit gave a partial result (e.g. gap=0, character=Metal), prefer that
    if vaspkit_result is not None:
        return vaspkit_result
    
    logger.error(f"All bandgap extraction methods failed for {material_id}")
    return default_result


def _parse_band_gap_file(band_gap_file: str, material_id: str) -> Dict[str, Any]:
    """Parse BAND_GAP file generated by vaspkit."""
    band_gap = None
    band_character = None
    vbm_location = None
    cbm_location = None
    is_spin_polarized = False
    
    with open(band_gap_file, 'r') as f:
        for line in f:
            if "Spin Channel:" in line:
                is_spin_polarized = True
            
            if "Band Character:" in line:
                band_character = line.split()[-1]
            elif "Band Gap (eV):" in line:
                band_gap = float(line.split()[-1])
            
            # Parse VBM/CBM locations
            if is_spin_polarized:
                if "Location of VBM (TO):" in line:
                    parts = line.split(':')[-1].strip().split()
                    vbm_location = tuple(float(x) for x in parts[:3])
                elif "Location of CBM (TO):" in line:
                    parts = line.split(':')[-1].strip().split()
                    cbm_location = tuple(float(x) for x in parts[:3])
            else:
                if "Location of VBM:" in line and "(UP)" not in line and "(DW)" not in line and "(TO)" not in line:
                    parts = line.split(':')[-1].strip().split()
                    vbm_location = tuple(float(x) for x in parts[:3])
                elif "Location of CBM:" in line and "(UP)" not in line and "(DW)" not in line and "(TO)" not in line:
                    parts = line.split(':')[-1].strip().split()
                    cbm_location = tuple(float(x) for x in parts[:3])
    
    # Determine band character for spin-polarized calculations
    if band_gap is not None and band_character is None and is_spin_polarized:
        band_character = _determine_band_character(
            band_gap, vbm_location, cbm_location, material_id
        )
    
    if band_gap is not None and band_character is not None:
        band_character = _normalize_band_character(
            band_character, vbm_location, cbm_location, material_id
        )
        
        logger.info(f"Result for {material_id}: {band_character}, {band_gap} eV")
        return {
            'id': material_id,
            'calculated_character': band_character,
            'calculated_gap': band_gap
        }
    
    logger.warning(f"Could not parse bandgap from BAND_GAP file for {material_id}")
    return {
        'id': material_id,
        'calculated_character': "ExtractionFailed",
        'calculated_gap': None,
        'extraction_error': "Could not parse BAND_GAP file"
    }


def _determine_band_character(
    band_gap: float,
    vbm_location: Optional[Tuple[float, float, float]],
    cbm_location: Optional[Tuple[float, float, float]],
    material_id: str
) -> str:
    """Determine band character from VBM/CBM locations."""
    if vbm_location is not None and cbm_location is not None:
        tolerance = 1e-4
        same_location = all(abs(v - c) < tolerance 
                          for v, c in zip(vbm_location, cbm_location))
        
        if band_gap < 0.01:
            logger.info(f"{material_id}: Spin-polarized with tiny gap ({band_gap:.4f} eV) -> Metallic")
            return "Metallic"
        elif same_location:
            logger.info(f"{material_id}: Spin-polarized, VBM=CBM location -> Direct")
            return "Direct"
        else:
            logger.info(f"{material_id}: Spin-polarized, VBM≠CBM location -> Indirect")
            return "Indirect"
    
    logger.warning(f"{material_id}: Could not parse VBM/CBM locations")
    return "Unknown"


def _normalize_band_character(
    band_character: str,
    vbm_location: Optional[Tuple[float, float, float]],
    cbm_location: Optional[Tuple[float, float, float]],
    material_id: str
) -> str:
    """Normalize band character and distinguish Metal vs Semi-metal."""
    if band_character.lower() in ["semimetallic", "semi-metallic"]:
        logger.info(f"{material_id}: Normalizing '{band_character}' to 'Metallic'")
        band_character = "Metallic"
    
    if band_character == "Metallic":
        if vbm_location is not None and cbm_location is not None:
            tolerance = 1e-6
            same_location = all(abs(v - c) < tolerance 
                              for v, c in zip(vbm_location, cbm_location))
            
            if same_location:
                band_character = "Metal"
                logger.info(f"{material_id}: VBM and CBM overlap -> Metal")
            else:
                band_character = "Semi-metal"
                logger.info(f"{material_id}: VBM and CBM at different k-points -> Semi-metal")
        else:
            band_character = "Metal"
            logger.warning(f"{material_id}: Could not parse VBM/CBM, defaulting to Metal")
    
    return band_character


def extract_bandgap_from_eigenval(
    eigenval_path: str,
    efermi: Optional[float] = None
) -> Optional[BandgapResult]:
    """
    Extract bandgap directly from EIGENVAL file.
    
    Uses an occupation-based approach that works for both ISPIN=1 and ISPIN=2:
    - VBM = max eigenvalue where occupation > 0.5
    - CBM = min eigenvalue where occupation < 0.5
    
    For ISPIN=2, handles two VASP EIGENVAL formats:
    1. Doubled k-point blocks: NKPTS blocks for spin-up, then NKPTS for spin-down
       (3 columns per band line: index, energy, occupation)
    2. Single k-point blocks with 5 columns: index, energy_up, occ_up, energy_down, occ_down
    
    Falls back to NELECT-based approach if occupations are not reliable.
    
    Args:
        eigenval_path: Path to EIGENVAL file
        efermi: Fermi energy (if known, used as fallback for VBM/CBM determination)
        
    Returns:
        BandgapResult or None if extraction fails
    """
    if not os.path.exists(eigenval_path):
        logger.warning(f"EIGENVAL not found: {eigenval_path}")
        return None
    
    try:
        with open(eigenval_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        # Line 1 (0-indexed: 0): NIONS, ?, ISPIN, ?
        header_line1 = lines[0].split()
        ispin = int(header_line1[2]) if len(header_line1) >= 3 else 1
        
        # Line 6 (0-indexed: 5): NELECT, NKPTS, NBANDS
        header = lines[5].split()
        nelect = int(float(header[0]))
        nkpts = int(header[1])
        nbands = int(header[2])
        
        logger.info(f"EIGENVAL: ISPIN={ispin}, NELECT={nelect}, NKPTS={nkpts}, NBANDS={nbands}")
        
        # Collect all (energy, occupation, k-point) tuples
        all_band_data = []  # List of (energy, occupation, kpt_tuple)
        
        line_idx = 7  # Start of k-point data
        
        # For ISPIN=2 with doubled blocks, total blocks = 2*NKPTS
        # For ISPIN=1 or ISPIN=2 with 5-column format, total blocks = NKPTS
        # Detect format from first band line
        
        # Find first band line to detect format
        temp_idx = line_idx
        while temp_idx < len(lines) and not lines[temp_idx].strip():
            temp_idx += 1
        temp_idx += 1  # Skip k-point line
        first_band_cols = len(lines[temp_idx].split()) if temp_idx < len(lines) else 3
        
        if ispin == 2 and first_band_cols == 3:
            # ISPIN=2 with doubled k-point blocks (standard VASP format)
            total_kblocks = 2 * nkpts
        else:
            total_kblocks = nkpts
        
        for ikpt in range(total_kblocks):
            # Skip empty lines
            while line_idx < len(lines) and not lines[line_idx].strip():
                line_idx += 1
            
            if line_idx >= len(lines):
                break
            
            # Read k-point
            kpt_line = lines[line_idx].split()
            kpt = (float(kpt_line[0]), float(kpt_line[1]), float(kpt_line[2]))
            line_idx += 1
            
            # Read band energies and occupations
            for iband in range(nbands):
                if line_idx >= len(lines):
                    break
                band_line = lines[line_idx].split()
                line_idx += 1
                
                if len(band_line) >= 3:
                    energy = float(band_line[1])
                    occ = float(band_line[2])
                    all_band_data.append((energy, occ, kpt))
                
                # Handle 5-column format (both spin channels in one line)
                if len(band_line) >= 5:
                    energy_dn = float(band_line[3])
                    occ_dn = float(band_line[4])
                    all_band_data.append((energy_dn, occ_dn, kpt))
        
        if not all_band_data:
            logger.warning(f"No band data extracted from EIGENVAL")
            return None
        
        # Occupation-based VBM/CBM detection
        # VBM = max energy where occupation > 0.5
        # CBM = min energy where occupation < 0.5
        OCC_THRESHOLD = 0.5
        
        occupied = [(e, kpt) for e, occ, kpt in all_band_data if occ > OCC_THRESHOLD]
        unoccupied = [(e, kpt) for e, occ, kpt in all_band_data if occ < OCC_THRESHOLD]
        
        if not occupied or not unoccupied:
            # Fallback: use NELECT-based approach for ISPIN=1
            if ispin == 1:
                logger.info(f"Occupation-based approach failed, using NELECT-based fallback")
                nocc = nelect // 2
                return _extract_bandgap_nelect_based(lines, nkpts, nbands, nocc)
            
            # For ISPIN=2 with no clear occupation boundary, try Fermi energy
            if efermi is not None:
                logger.info(f"Using Fermi energy ({efermi:.4f} eV) for VBM/CBM detection")
                below_fermi = [(e, kpt) for e, occ, kpt in all_band_data if e <= efermi + 0.1]
                above_fermi = [(e, kpt) for e, occ, kpt in all_band_data if e >= efermi - 0.1]
                
                if below_fermi and above_fermi:
                    vbm_energy, vbm_kpt = max(below_fermi, key=lambda x: x[0])
                    cbm_energy, cbm_kpt = min(above_fermi, key=lambda x: x[0])
                    
                    if cbm_energy > vbm_energy:
                        bandgap = cbm_energy - vbm_energy
                    else:
                        bandgap = 0.0
                    
                    is_direct = _kpoints_match(vbm_kpt, cbm_kpt)
                    is_metal = bandgap < 0.01
                    
                    return BandgapResult(
                        bandgap=max(0.0, bandgap),
                        is_direct=is_direct,
                        vbm_energy=vbm_energy,
                        cbm_energy=cbm_energy,
                        vbm_kpoint=vbm_kpt,
                        cbm_kpoint=cbm_kpt,
                        is_metal=is_metal,
                        spin_polarized=(ispin == 2),
                        fermi_energy=efermi,
                    )
            
            logger.warning(f"Cannot determine VBM/CBM from EIGENVAL")
            return None
        
        # Find VBM and CBM
        vbm_energy, vbm_kpt = max(occupied, key=lambda x: x[0])
        cbm_energy, cbm_kpt = min(unoccupied, key=lambda x: x[0])
        
        bandgap = cbm_energy - vbm_energy
        is_direct = _kpoints_match(vbm_kpt, cbm_kpt)
        is_metal = bandgap < 0.01
        
        logger.info(f"EIGENVAL result: VBM={vbm_energy:.4f} eV at {vbm_kpt}, "
                     f"CBM={cbm_energy:.4f} eV at {cbm_kpt}, gap={bandgap:.4f} eV")
        
        return BandgapResult(
            bandgap=max(0.0, bandgap),
            is_direct=is_direct,
            vbm_energy=vbm_energy,
            cbm_energy=cbm_energy,
            vbm_kpoint=vbm_kpt,
            cbm_kpoint=cbm_kpt,
            is_metal=is_metal,
            spin_polarized=(ispin == 2),
            fermi_energy=efermi,
        )
        
    except Exception as e:
        logger.error(f"Failed to parse EIGENVAL: {e}")
        return None


def _kpoints_match(
    kpt1: Tuple[float, float, float],
    kpt2: Tuple[float, float, float],
    tolerance: float = 1e-4
) -> bool:
    """Check if two k-points are the same within tolerance."""
    return all(abs(a - b) < tolerance for a, b in zip(kpt1, kpt2))


def _extract_bandgap_nelect_based(
    lines: List[str],
    nkpts: int,
    nbands: int,
    nocc: int
) -> Optional[BandgapResult]:
    """
    Fallback EIGENVAL parsing using NELECT-based band counting.
    Only works for ISPIN=1.
    """
    vbm = float('-inf')
    cbm = float('inf')
    vbm_kpt = (0.0, 0.0, 0.0)
    cbm_kpt = (0.0, 0.0, 0.0)
    
    line_idx = 7
    
    for ikpt in range(nkpts):
        while line_idx < len(lines) and not lines[line_idx].strip():
            line_idx += 1
        
        kpt_line = lines[line_idx].split()
        kpt = (float(kpt_line[0]), float(kpt_line[1]), float(kpt_line[2]))
        line_idx += 1
        
        energies = []
        for iband in range(nbands):
            band_line = lines[line_idx].split()
            energies.append(float(band_line[1]))
            line_idx += 1
        
        if nocc > 0 and nocc < len(energies):
            if energies[nocc - 1] > vbm:
                vbm = energies[nocc - 1]
                vbm_kpt = kpt
            if energies[nocc] < cbm:
                cbm = energies[nocc]
                cbm_kpt = kpt
    
    if vbm == float('-inf') or cbm == float('inf'):
        return None
    
    bandgap = cbm - vbm
    is_direct = _kpoints_match(vbm_kpt, cbm_kpt)
    is_metal = bandgap < 0.01
    
    return BandgapResult(
        bandgap=max(0.0, bandgap),
        is_direct=is_direct,
        vbm_energy=vbm,
        cbm_energy=cbm,
        vbm_kpoint=vbm_kpt,
        cbm_kpoint=cbm_kpt,
        is_metal=is_metal
    )


def classify_electronic_type(
    vbm_energy: float,
    cbm_energy: float,
    fermi_energy: float,
    bandgap: Optional[float] = None,
    threshold_intrinsic: float = 0.1,
    threshold_metal: float = 0.01
) -> ElectronicTypeResult:
    """
    Classify electronic type based on Fermi level position relative to VBM/CBM.
    
    根据费米能级相对于 VBM/CBM 的位置判断材料的导电类型。
    
    Classification logic:
    - Metal: bandgap < threshold_metal (essentially zero)
    - n-type: E_f closer to CBM (within bandgap, but E_f - midgap > threshold)
    - p-type: E_f closer to VBM (within bandgap, but midgap - E_f > threshold)
    - intrinsic: E_f near the middle of the bandgap
    
    Args:
        vbm_energy: Valence band maximum energy (eV)
        cbm_energy: Conduction band minimum energy (eV)
        fermi_energy: Fermi energy (eV)
        bandgap: Band gap in eV (if None, calculated from CBM-VBM)
        threshold_intrinsic: Threshold for intrinsic classification (fraction of bandgap)
        threshold_metal: Threshold below which material is considered metallic (eV)
        
    Returns:
        ElectronicTypeResult with classification and details
    """
    # 计算带隙
    if bandgap is None:
        bandgap = cbm_energy - vbm_energy
    
    # 计算费米能级与 VBM/CBM 的距离
    fermi_to_vbm = fermi_energy - vbm_energy
    fermi_to_cbm = cbm_energy - fermi_energy
    
    # 初始化结果
    electronic_type = "unknown"
    confidence = "low"
    notes = ""
    
    # 判断逻辑
    if bandgap < threshold_metal:
        # 金属：带隙几乎为零
        electronic_type = "metal"
        confidence = "high"
        notes = f"Metallic (bandgap={bandgap:.4f} eV < {threshold_metal} eV)"
        logger.info(f"Electronic type: metal (bandgap={bandgap:.4f} eV)")
        
    elif fermi_energy > cbm_energy:
        # 费米能级在导带之上：重掺杂 n-type 或退化半导体
        electronic_type = "n-type"
        confidence = "high"
        notes = f"Heavily doped n-type (E_f above CBM by {fermi_energy - cbm_energy:.3f} eV)"
        logger.info(f"Electronic type: n-type (degenerate, E_f > CBM)")
        
    elif fermi_energy < vbm_energy:
        # 费米能级在价带之下：重掺杂 p-type 或退化半导体
        electronic_type = "p-type"
        confidence = "high"
        notes = f"Heavily doped p-type (E_f below VBM by {vbm_energy - fermi_energy:.3f} eV)"
        logger.info(f"Electronic type: p-type (degenerate, E_f < VBM)")
        
    else:
        # 费米能级在带隙内
        midgap = (vbm_energy + cbm_energy) / 2.0
        deviation_from_midgap = fermi_energy - midgap
        relative_deviation = abs(deviation_from_midgap) / bandgap if bandgap > 0 else 0
        
        if relative_deviation < threshold_intrinsic:
            # 费米能级接近带隙中心：本征半导体/绝缘体
            electronic_type = "intrinsic"
            confidence = "high" if relative_deviation < threshold_intrinsic / 2 else "medium"
            notes = f"Intrinsic (E_f at {50 + 100*deviation_from_midgap/bandgap:.1f}% of bandgap)"
            logger.info(f"Electronic type: intrinsic (E_f near midgap)")
            
        elif deviation_from_midgap > 0:
            # 费米能级偏向导带：n-type
            electronic_type = "n-type"
            proximity_to_cbm = fermi_to_cbm / bandgap
            if proximity_to_cbm < 0.2:
                confidence = "high"
            elif proximity_to_cbm < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            notes = f"n-type (E_f at {100*fermi_to_vbm/bandgap:.1f}% above VBM)"
            logger.info(f"Electronic type: n-type (E_f closer to CBM)")
            
        else:
            # 费米能级偏向价带：p-type
            electronic_type = "p-type"
            proximity_to_vbm = fermi_to_vbm / bandgap
            if proximity_to_vbm < 0.2:
                confidence = "high"
            elif proximity_to_vbm < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            notes = f"p-type (E_f at {100*fermi_to_vbm/bandgap:.1f}% above VBM)"
            logger.info(f"Electronic type: p-type (E_f closer to VBM)")
    
    return ElectronicTypeResult(
        electronic_type=electronic_type,
        fermi_energy=fermi_energy,
        vbm_energy=vbm_energy,
        cbm_energy=cbm_energy,
        bandgap=bandgap,
        fermi_to_vbm=fermi_to_vbm,
        fermi_to_cbm=fermi_to_cbm,
        confidence=confidence,
        notes=notes
    )


def extract_electronic_type_from_calculation(
    calc_dir: str,
    material_id: str = ""
) -> Optional[ElectronicTypeResult]:
    """
    Extract electronic type from a VASP calculation directory.
    
    从 VASP 计算目录中提取电子类型分类信息。
    需要 EIGENVAL 和 OUTCAR 文件。
    
    Args:
        calc_dir: Directory containing VASP outputs (EIGENVAL, OUTCAR)
        material_id: Material identifier for logging
        
    Returns:
        ElectronicTypeResult or None if extraction fails
    """
    eigenval_path = os.path.join(calc_dir, 'EIGENVAL')
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    
    # 首先提取带隙和 VBM/CBM
    bandgap_result = extract_bandgap_from_eigenval(eigenval_path)
    if bandgap_result is None:
        logger.warning(f"Could not extract bandgap from {calc_dir}")
        return None
    
    # 从 OUTCAR 提取费米能级
    fermi_energy = _extract_fermi_from_outcar(outcar_path)
    if fermi_energy is None:
        logger.warning(f"Could not extract Fermi energy from {outcar_path}")
        # 如果无法获取费米能级，假设它在 VBM 处（非掺杂情况）
        fermi_energy = bandgap_result.vbm_energy
        logger.info(f"Assuming Fermi energy at VBM: {fermi_energy:.4f} eV")
    
    # 分类电子类型
    result = classify_electronic_type(
        vbm_energy=bandgap_result.vbm_energy,
        cbm_energy=bandgap_result.cbm_energy,
        fermi_energy=fermi_energy,
        bandgap=bandgap_result.bandgap
    )
    
    logger.info(f"[{material_id}] Electronic type: {result.electronic_type} "
                f"(confidence: {result.confidence})")
    
    return result


def _extract_fermi_from_outcar(outcar_path: str) -> Optional[float]:
    """
    Extract Fermi energy from OUTCAR file.
    
    从 OUTCAR 文件提取费米能级。
    
    Args:
        outcar_path: Path to OUTCAR file
        
    Returns:
        Fermi energy in eV, or None if not found
    """
    if not os.path.exists(outcar_path):
        return None
    
    try:
        # Use reversed() to get the LAST E-fermi value (the converged one).
        # In relaxation OUTCARs (NSW>0), E-fermi appears once per ionic step;
        # the first is the initial (unconverged) value, the last is correct.
        # For static/band calculations (NSW=0) there is only one, so this is safe.
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if 'E-fermi' in line:
                # Format: " E-fermi :   5.1234     XC(G=0):  -9.1234     alpha+bet : -7.1234"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'E-fermi' and i + 2 < len(parts):
                        return float(parts[i + 2])
    except Exception as e:
        logger.warning(f"Failed to extract Fermi energy from OUTCAR: {e}")
    
    return None
