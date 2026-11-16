"""
Phonon and vibrational spectra extraction module.

提供声子频率和光谱的提取功能：
- 从 DFPT 计算中提取声子频率
- 拉曼光谱和活性
- 红外光谱和强度
- 热力学性质计算（振动熵、自由能等）

Provides functions to extract:
- Phonon frequencies from DFPT calculations
- Raman spectra and activities
- IR spectra and intensities
- Thermodynamic properties (vibrational entropy, free energy, etc.)
"""
import os
import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# 物理常数
# Physical constants
KB_EV = 8.617333262e-5  # Boltzmann constant in eV/K
HBAR_EV_S = 6.582119569e-16  # Reduced Planck constant in eV·s
CM_TO_EV = 1.23984198e-4  # cm^-1 to eV conversion factor
THZ_TO_EV = 4.135667696e-3  # THz to eV conversion factor
MEV_TO_EV = 1e-3  # meV to eV


@dataclass
class PhononResult:
    """Container for phonon calculation results."""
    # Phonon frequencies (cm^-1)
    frequencies: List[float] = field(default_factory=list)
    # Frequency in THz
    frequencies_THz: List[float] = field(default_factory=list)
    # Frequency in meV
    frequencies_meV: List[float] = field(default_factory=list)
    
    # Mode information
    mode_indices: List[int] = field(default_factory=list)
    is_imaginary: List[bool] = field(default_factory=list)  # Imaginary (unstable) modes
    
    # Raman data
    raman_activities: List[float] = field(default_factory=list)  # A^4/amu
    raman_frequencies: List[float] = field(default_factory=list)  # cm^-1
    
    # IR data
    ir_intensities: List[float] = field(default_factory=list)  # km/mol
    ir_frequencies: List[float] = field(default_factory=list)  # cm^-1
    
    # Eigenvectors (mode displacements)
    eigenvectors: Optional[np.ndarray] = None  # Shape: (n_modes, n_atoms, 3)
    
    # Metadata
    n_atoms: int = 0
    n_modes: int = 0
    has_raman: bool = False
    has_ir: bool = False
    calculation_type: str = ""  # "DFPT" or "finite_displacement"


@dataclass
class ThermodynamicResult:
    """
    Container for thermodynamic properties from phonon calculations.
    
    从声子计算得到的热力学性质容器。
    所有能量单位为 eV/cell，熵单位为 eV/K/cell。
    """
    temperature: float  # Temperature (K)
    
    # 热力学量 / Thermodynamic quantities (per unit cell)
    vibrational_entropy: float  # S_vib (eV/K)
    helmholtz_free_energy: float  # F = U - TS (eV)
    internal_energy: float  # U (eV)
    heat_capacity_cv: float  # Cv (eV/K)
    zero_point_energy: float  # ZPE (eV)
    
    # 各模式贡献 / Per-mode contributions
    mode_contributions: Optional[Dict[str, np.ndarray]] = None
    
    # 元数据 / Metadata
    n_modes: int = 0
    n_imaginary: int = 0  # Number of imaginary modes (excluded from calculation)
    notes: str = ""


def extract_phonon_frequencies(calc_dir: str) -> Optional[PhononResult]:
    """
    Extract phonon frequencies from VASP DFPT or finite displacement calculation.
    
    Args:
        calc_dir: Directory containing OUTCAR from DFPT/phonon calculation
        
    Returns:
        PhononResult object or None
    """
    outcar_path = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.exists(outcar_path):
        logger.warning(f"OUTCAR not found in {calc_dir}")
        return None
    
    result = PhononResult()
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Determine calculation type from INCAR parameters in OUTCAR
        if 'IBRION =      7' in content or 'IBRION =      8' in content:
            result.calculation_type = "DFPT"
        elif 'IBRION =      5' in content or 'IBRION =      6' in content:
            result.calculation_type = "finite_displacement"
        
        # Extract frequencies using regex
        # Pattern: "   N f  =    X.XXXXX THz    X.XXXXX 2PiTHz   X.XXXXX cm-1    X.XXXXX meV"
        freq_pattern = re.compile(
            r'^\s*(\d+)\s+f\s*=\s*([\d.]+)\s+THz\s+([\d.]+)\s+2PiTHz\s+([\d.]+)\s+cm-1\s+([\d.]+)\s+meV',
            re.MULTILINE
        )
        # Pattern for imaginary frequencies
        imag_pattern = re.compile(
            r'^\s*(\d+)\s+f/i\s*=\s*([\d.]+)\s+THz\s+([\d.]+)\s+2PiTHz\s+([\d.]+)\s+cm-1\s+([\d.]+)\s+meV',
            re.MULTILINE
        )
        
        frequencies = {}  # mode_num -> (THz, cm-1, meV, is_imaginary)
        
        for match in freq_pattern.finditer(content):
            mode_num = int(match.group(1))
            if mode_num not in frequencies:  # Only first occurrence
                frequencies[mode_num] = {
                    'THz': float(match.group(2)),
                    'cm-1': float(match.group(4)),
                    'meV': float(match.group(5)),
                    'imaginary': False
                }
        
        for match in imag_pattern.finditer(content):
            mode_num = int(match.group(1))
            if mode_num not in frequencies:
                frequencies[mode_num] = {
                    'THz': float(match.group(2)),
                    'cm-1': float(match.group(4)),
                    'meV': float(match.group(5)),
                    'imaginary': True
                }
        
        # Sort by mode number and populate result
        for mode_num in sorted(frequencies.keys()):
            data = frequencies[mode_num]
            result.mode_indices.append(mode_num)
            result.frequencies.append(data['cm-1'])
            result.frequencies_THz.append(data['THz'])
            result.frequencies_meV.append(data['meV'])
            result.is_imaginary.append(data['imaginary'])
        
        result.n_modes = len(result.frequencies)
        
        # Try to extract number of atoms
        natoms_match = re.search(r'NIONS\s*=\s*(\d+)', content)
        if natoms_match:
            result.n_atoms = int(natoms_match.group(1))
        
        logger.info(f"Extracted {result.n_modes} phonon modes from {result.calculation_type} calculation")
        n_imag = sum(result.is_imaginary)
        if n_imag > 0:
            logger.warning(f"  {n_imag} imaginary modes detected!")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract phonon frequencies: {e}")
        return None


def extract_raman_spectra(calc_dir: str, material_id: str = "") -> PhononResult:
    """
    Extract Raman spectra data from DFPT calculation.
    
    Checks for RAMAN.dat file (from vaspkit) or parses OUTCAR directly.
    
    Args:
        calc_dir: Directory containing calculation outputs
        material_id: Material identifier for logging
        
    Returns:
        PhononResult with Raman data
    """
    result = PhononResult()
    
    # First try RAMAN.dat from vaspkit
    raman_file = os.path.join(calc_dir, 'RAMAN.dat')
    if os.path.exists(raman_file):
        result = _parse_raman_file(raman_file)
        if result.has_raman:
            logger.info(f"Extracted {len(result.raman_frequencies)} Raman-active modes from RAMAN.dat")
            return result
    
    # Fall back to OUTCAR parsing
    phonon_result = extract_phonon_frequencies(calc_dir)
    if phonon_result is not None:
        result = phonon_result
        
        # Filter optical modes (exclude acoustic near zero)
        acoustic_threshold = 10.0  # cm^-1
        for i, freq in enumerate(result.frequencies):
            if freq > acoustic_threshold and not result.is_imaginary[i]:
                result.raman_frequencies.append(freq)
                result.raman_activities.append(0.0)  # Placeholder
        
        if result.raman_frequencies:
            result.has_raman = True
    
    return result


def _parse_raman_file(raman_file: str) -> PhononResult:
    """Parse RAMAN.dat file generated by vaspkit."""
    result = PhononResult()
    
    try:
        with open(raman_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    activity = float(parts[1])
                    if freq > 0:  # Exclude imaginary modes
                        result.raman_frequencies.append(freq)
                        result.raman_activities.append(activity)
                except ValueError:
                    continue
        
        if result.raman_frequencies:
            result.has_raman = True
            result.n_modes = len(result.raman_frequencies)
        
        return result
        
    except Exception as e:
        logger.warning(f"Failed to parse RAMAN.dat: {e}")
        return result


def extract_ir_spectra(calc_dir: str) -> PhononResult:
    """
    Extract IR spectra data from DFPT calculation.
    
    Args:
        calc_dir: Directory containing calculation outputs
        
    Returns:
        PhononResult with IR data
    """
    result = PhononResult()
    
    # Try IR.dat from vaspkit
    ir_file = os.path.join(calc_dir, 'IR.dat')
    if os.path.exists(ir_file):
        try:
            with open(ir_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        freq = float(parts[0])
                        intensity = float(parts[1])
                        if freq > 0:
                            result.ir_frequencies.append(freq)
                            result.ir_intensities.append(intensity)
                    except ValueError:
                        continue
            
            if result.ir_frequencies:
                result.has_ir = True
                logger.info(f"Extracted {len(result.ir_frequencies)} IR-active modes")
            
        except Exception as e:
            logger.warning(f"Failed to parse IR.dat: {e}")
    
    return result


def run_vaspkit_raman(calc_dir: str) -> PhononResult:
    """
    Run vaspkit to extract Raman spectra.
    
    Args:
        calc_dir: Directory containing VASP outputs
        
    Returns:
        PhononResult with Raman data
    """
    from execution.job_manager import run_command
    
    try:
        # Task 502: Extract Raman spectra
        run_command("vaspkit -task 502", cwd=calc_dir, shell=True)
        
        # Parse the generated file
        return extract_raman_spectra(calc_dir)
        
    except Exception as e:
        logger.error(f"Failed to run vaspkit Raman extraction: {e}")
        return PhononResult()


def check_material_suitability_for_raman(
    calc_dir: str,
    is_metal: bool = False,
    is_magnetic: bool = False
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if material is suitable for Raman calculation.
    
    Args:
        calc_dir: Directory containing structure/OUTCAR
        is_metal: Whether material is metallic
        is_magnetic: Whether material is magnetic
        
    Returns:
        Tuple of (is_suitable, reason, recommended_settings)
    """
    recommended = {}
    
    if is_metal:
        return (
            False,
            "Material is metallic - Raman calculation not applicable (DFPT diverges for metals)",
            {}
        )
    
    if is_magnetic:
        # Can still do phonons, but must use finite displacement
        recommended = {
            'IBRION': 5,  # Central finite differences
            'POTIM': 0.015,
            'note': 'DFPT does not support ISPIN=2, using finite displacement method'
        }
        return (
            True,
            "Magnetic material - using finite displacement method instead of DFPT",
            recommended
        )
    
    # Non-magnetic insulator/semiconductor - DFPT is optimal
    recommended = {
        'IBRION': 8,
        'LEPSILON': True,
        'note': 'DFPT method for non-magnetic insulator'
    }
    return (
        True,
        "Material suitable for DFPT Raman calculation",
        recommended
    )


def calculate_thermodynamic_properties(
    phonon_result: PhononResult,
    temperature: float = 300.0,
    exclude_acoustic: bool = True,
    acoustic_threshold: float = 5.0
) -> ThermodynamicResult:
    """
    Calculate thermodynamic properties from phonon frequencies.
    
    从声子频率计算热力学性质，使用谐振子近似。
    
    Formulas (quantum harmonic oscillator):
    - Zero-point energy: ZPE = (1/2) * sum(hbar * omega)
    - Internal energy: U = ZPE + sum(hbar * omega * n(omega, T))
    - Helmholtz free energy: F = U - T*S
    - Vibrational entropy: S = kB * sum[x/(exp(x)-1) - ln(1-exp(-x))]
      where x = hbar * omega / (kB * T)
    - Heat capacity: Cv = kB * sum[x^2 * exp(x) / (exp(x)-1)^2]
    
    Note: Imaginary frequencies and acoustic modes near Gamma are excluded.
    
    Args:
        phonon_result: PhononResult containing frequencies
        temperature: Temperature in Kelvin
        exclude_acoustic: Whether to exclude acoustic modes (near zero frequency)
        acoustic_threshold: Frequency threshold (cm^-1) below which modes are acoustic
        
    Returns:
        ThermodynamicResult with all thermodynamic quantities
    """
    if not phonon_result.frequencies:
        logger.warning("No phonon frequencies available for thermodynamic calculation")
        return ThermodynamicResult(
            temperature=temperature,
            vibrational_entropy=0.0,
            helmholtz_free_energy=0.0,
            internal_energy=0.0,
            heat_capacity_cv=0.0,
            zero_point_energy=0.0,
            notes="No frequencies available"
        )
    
    # 获取频率（以 eV 为单位）
    frequencies_cm = np.array(phonon_result.frequencies)
    is_imaginary = np.array(phonon_result.is_imaginary) if phonon_result.is_imaginary else np.zeros(len(frequencies_cm), dtype=bool)
    
    # 转换 cm^-1 到 eV
    frequencies_ev = frequencies_cm * CM_TO_EV
    
    # 过滤掉虚频和声学模式
    valid_mask = ~is_imaginary
    if exclude_acoustic:
        valid_mask &= (frequencies_cm > acoustic_threshold)
    
    valid_freqs = frequencies_ev[valid_mask]
    n_valid = len(valid_freqs)
    n_imaginary = np.sum(is_imaginary)
    n_acoustic = np.sum((~is_imaginary) & (frequencies_cm <= acoustic_threshold))
    
    if n_valid == 0:
        logger.warning("No valid optical modes for thermodynamic calculation")
        return ThermodynamicResult(
            temperature=temperature,
            vibrational_entropy=0.0,
            helmholtz_free_energy=0.0,
            internal_energy=0.0,
            heat_capacity_cv=0.0,
            zero_point_energy=0.0,
            n_modes=0,
            n_imaginary=n_imaginary,
            notes=f"No valid modes (excluded {n_imaginary} imaginary, {n_acoustic} acoustic)"
        )
    
    logger.info(f"Calculating thermodynamics at T={temperature} K "
                f"with {n_valid} modes (excluded {n_imaginary} imaginary, "
                f"{n_acoustic} acoustic)")
    
    # 计算零点能 (Zero-Point Energy)
    # ZPE = (1/2) * sum(hbar * omega)
    zpe = 0.5 * np.sum(valid_freqs)
    
    # 计算玻色-爱因斯坦分布函数
    # n(omega, T) = 1 / (exp(hbar*omega/kB*T) - 1)
    if temperature > 0:
        # x = hbar * omega / (kB * T) = omega_eV / (kB_eV * T)
        x = valid_freqs / (KB_EV * temperature)
        
        # 避免数值溢出
        x_safe = np.clip(x, 0, 700)  # exp(700) ~ 1e304
        
        # 玻色分布
        exp_x = np.exp(x_safe)
        n_bose = 1.0 / (exp_x - 1.0)
        
        # 内能：U = ZPE + sum(hbar * omega * n)
        internal_energy = zpe + np.sum(valid_freqs * n_bose)
        
        # 熵：S = kB * sum[x*n - ln(1 - exp(-x))]
        # 另一形式：S = kB * sum[x/(exp(x)-1) - ln(1-exp(-x))]
        entropy_per_mode = x * n_bose - np.log(1.0 - np.exp(-x_safe))
        vibrational_entropy = KB_EV * np.sum(entropy_per_mode)
        
        # 热容：Cv = kB * sum[x^2 * exp(x) / (exp(x)-1)^2]
        cv_per_mode = x_safe**2 * exp_x / (exp_x - 1.0)**2
        heat_capacity = KB_EV * np.sum(cv_per_mode)
        
        # 亥姆霍兹自由能：F = U - T*S
        helmholtz_free_energy = internal_energy - temperature * vibrational_entropy
        
        # 记录各模式贡献
        mode_contributions = {
            'entropy': entropy_per_mode * KB_EV,
            'heat_capacity': cv_per_mode * KB_EV,
            'bose_occupation': n_bose,
        }
        
    else:
        # T = 0K 的极限
        internal_energy = zpe
        vibrational_entropy = 0.0
        heat_capacity = 0.0
        helmholtz_free_energy = zpe
        mode_contributions = None
    
    notes = (f"Calculated from {n_valid} optical modes at T={temperature} K. "
             f"Excluded: {n_imaginary} imaginary, {n_acoustic} acoustic.")
    
    logger.info(f"Thermodynamic results: ZPE={zpe:.4f} eV, "
                f"F={helmholtz_free_energy:.4f} eV, "
                f"S={vibrational_entropy:.6f} eV/K, "
                f"Cv={heat_capacity:.6f} eV/K")
    
    return ThermodynamicResult(
        temperature=temperature,
        vibrational_entropy=vibrational_entropy,
        helmholtz_free_energy=helmholtz_free_energy,
        internal_energy=internal_energy,
        heat_capacity_cv=heat_capacity,
        zero_point_energy=zpe,
        mode_contributions=mode_contributions,
        n_modes=n_valid,
        n_imaginary=n_imaginary,
        notes=notes
    )


def calculate_thermodynamic_properties_range(
    phonon_result: PhononResult,
    temperatures: List[float] = None,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    t_step: float = 50.0
) -> List[ThermodynamicResult]:
    """
    Calculate thermodynamic properties over a temperature range.
    
    在温度范围内计算热力学性质。
    
    Args:
        phonon_result: PhononResult containing frequencies
        temperatures: List of temperatures (K). If None, use t_min/t_max/t_step.
        t_min: Minimum temperature (K)
        t_max: Maximum temperature (K)
        t_step: Temperature step (K)
        
    Returns:
        List of ThermodynamicResult for each temperature
    """
    if temperatures is None:
        temperatures = np.arange(t_min, t_max + t_step/2, t_step).tolist()
        # 确保 0K 用一个非常小的正数代替，避免除零
        temperatures = [max(0.1, t) for t in temperatures]
    
    results = []
    for temp in temperatures:
        result = calculate_thermodynamic_properties(phonon_result, temperature=temp)
        results.append(result)
    
    logger.info(f"Calculated thermodynamics for {len(results)} temperatures "
                f"from {temperatures[0]:.1f} K to {temperatures[-1]:.1f} K")
    
    return results


def extract_thermodynamic_from_calculation(
    calc_dir: str,
    temperature: float = 300.0
) -> Optional[ThermodynamicResult]:
    """
    Extract thermodynamic properties from a VASP phonon calculation.
    
    从 VASP 声子计算目录中提取热力学性质。
    
    Args:
        calc_dir: Directory containing OUTCAR from phonon calculation
        temperature: Temperature for thermodynamic calculation (K)
        
    Returns:
        ThermodynamicResult or None if extraction fails
    """
    # 首先提取声子频率
    phonon_result = extract_phonon_frequencies(calc_dir)
    
    if phonon_result is None or phonon_result.n_modes == 0:
        logger.warning(f"Could not extract phonon frequencies from {calc_dir}")
        return None
    
    # 计算热力学性质
    thermo_result = calculate_thermodynamic_properties(
        phonon_result,
        temperature=temperature
    )
    
    return thermo_result


def get_gibbs_free_energy(
    helmholtz_free_energy: float,
    pressure: float = 0.0,
    volume: float = 0.0
) -> float:
    """
    Calculate Gibbs free energy from Helmholtz free energy.
    
    从亥姆霍兹自由能计算吉布斯自由能。
    
    G = F + PV
    
    Note: For solids at ambient pressure, PV term is typically negligible.
    
    Args:
        helmholtz_free_energy: Helmholtz free energy (eV)
        pressure: Pressure (GPa)
        volume: Volume (Å³)
        
    Returns:
        Gibbs free energy (eV)
    """
    # 转换 GPa·Å³ 到 eV: 1 GPa·Å³ = 0.00624150913 eV
    pv_term = pressure * volume * 0.00624150913
    return helmholtz_free_energy + pv_term
