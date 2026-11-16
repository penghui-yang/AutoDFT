"""
Charged defect calculation support module.

提供带电缺陷计算的支持功能：
- 从自然语言解析电荷状态
- 计算 NELECT 参数
- 生成缺陷计算所需的 INCAR 设置

Provides support for charged defect calculations:
- Parse charge state from natural language
- Calculate NELECT parameter
- Generate INCAR settings for defect calculations
"""
import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChargedDefectSettings:
    """
    Settings for charged defect calculation.
    
    带电缺陷计算的设置参数
    """
    charge_state: int  # Charge state (e.g., -1, 0, +1, +2)
    nelect: int  # Total number of electrons
    nelect_neutral: int  # Number of electrons for neutral system
    incar_settings: Dict[str, Any]  # Additional INCAR settings
    notes: str = ""  # Additional notes


def parse_charge_state_from_text(text: str) -> Optional[int]:
    """
    Parse charge state from natural language task description.
    
    从自然语言任务描述中解析电荷状态。
    
    Supported formats (examples):
    - "charge state +2" or "charge state -1"
    - "charged +2" or "charged -1"
    - "+2 charge" or "-1 charge"
    - "带 +2 电荷" or "带 -1 价电荷"
    - "q=+1" or "q=-2"
    - "defect with charge 2+" or "defect with charge 1-"
    - "positively charged" (assumes +1)
    - "negatively charged" (assumes -1)
    - "neutral" or "q=0" (returns 0)
    
    Args:
        text: Natural language text containing charge information
        
    Returns:
        Integer charge state, or None if not found
    """
    text_lower = text.lower()
    
    # 模式列表：(正则表达式, 提取函数)
    patterns = [
        # "charge state +2", "charge state -1", "charged +2"
        (r'charge[d]?\s*(?:state)?\s*([+-]?\d+)', lambda m: int(m.group(1))),
        # "+2 charge", "-1 charge"
        (r'([+-]?\d+)\s*charge', lambda m: int(m.group(1))),
        # "q=+1", "q=-2", "q = 0"
        (r'q\s*=\s*([+-]?\d+)', lambda m: int(m.group(1))),
        # "charge 2+" or "charge 1-" (chemistry notation)
        (r'charge\s*(\d+)([+-])', lambda m: int(m.group(1)) * (1 if m.group(2) == '+' else -1)),
        # "2+ charge" or "1- charge"
        (r'(\d+)([+-])\s*charge', lambda m: int(m.group(1)) * (1 if m.group(2) == '+' else -1)),
        # "带 +2 电荷" or "带 -1 价电荷" (Chinese)
        (r'带\s*([+-]?\d+)\s*(?:价)?电荷', lambda m: int(m.group(1))),
        # "+2 charged" or "-1 charged"
        (r'([+-]\d+)\s*charged', lambda m: int(m.group(1))),
        # "NELECT = X" (direct specification)
        # This is handled separately as it's the total electron count, not charge
    ]
    
    for pattern, extractor in patterns:
        match = re.search(pattern, text_lower)
        if match:
            charge = extractor(match)
            logger.info(f"Parsed charge state: {charge:+d} from pattern '{pattern}'")
            return charge
    
    # 检查定性描述
    if 'positively charged' in text_lower or 'positive charge' in text_lower:
        logger.info("Parsed charge state: +1 (from 'positively charged')")
        return 1
    if 'negatively charged' in text_lower or 'negative charge' in text_lower:
        logger.info("Parsed charge state: -1 (from 'negatively charged')")
        return -1
    if 'neutral' in text_lower and 'defect' in text_lower:
        logger.info("Parsed charge state: 0 (neutral defect)")
        return 0
    
    # 没有找到电荷信息
    return None


def calculate_nelect_from_potcar(potcar_path: str) -> Optional[int]:
    """
    Calculate total ZVAL (number of valence electrons) from POTCAR.
    
    从 POTCAR 文件计算总价电子数。
    
    Args:
        potcar_path: Path to POTCAR file
        
    Returns:
        Total number of valence electrons, or None if parsing fails
    """
    if not os.path.exists(potcar_path):
        logger.warning(f"POTCAR not found: {potcar_path}")
        return None
    
    try:
        total_zval = 0.0
        atom_counts = []  # 需要从 POSCAR 获取
        zvals = []
        
        with open(potcar_path, 'r') as f:
            for line in f:
                # ZVAL 行格式: "   ZVAL   =  6.0000    mass and calculation ..."
                if 'ZVAL' in line and '=' in line:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        zval_str = parts[1].split()[0]
                        zvals.append(float(zval_str))
        
        logger.info(f"Found ZVAL values: {zvals}")
        return zvals
        
    except Exception as e:
        logger.error(f"Failed to parse POTCAR for ZVAL: {e}")
        return None


def calculate_nelect(
    poscar_path: str,
    potcar_path: str,
    charge_state: int = 0
) -> Optional[ChargedDefectSettings]:
    """
    Calculate NELECT for a charged system.
    
    计算带电体系的总电子数 NELECT。
    
    NELECT = sum(n_i * ZVAL_i) - charge
    where:
    - n_i is the number of atoms of species i
    - ZVAL_i is the number of valence electrons for species i
    - charge is the total charge of the system (positive = electron deficient)
    
    Args:
        poscar_path: Path to POSCAR file
        potcar_path: Path to POTCAR file
        charge_state: Charge of the system (e.g., +1 means 1 electron removed)
        
    Returns:
        ChargedDefectSettings with NELECT and related settings
    """
    # 解析 POSCAR 获取原子数
    if not os.path.exists(poscar_path):
        logger.error(f"POSCAR not found: {poscar_path}")
        return None
    
    try:
        with open(poscar_path, 'r') as f:
            lines = f.readlines()
        
        # 第 6 行：元素名称
        # 第 7 行：各元素原子数
        atom_counts = [int(x) for x in lines[6].split()]
        logger.info(f"Atom counts from POSCAR: {atom_counts}")
        
    except Exception as e:
        logger.error(f"Failed to parse POSCAR: {e}")
        return None
    
    # 从 POTCAR 获取 ZVAL
    zvals = calculate_nelect_from_potcar(potcar_path)
    if zvals is None:
        return None
    
    if len(zvals) != len(atom_counts):
        logger.error(f"Mismatch: {len(zvals)} ZVAL values but {len(atom_counts)} atom types")
        return None
    
    # 计算中性体系的总电子数
    nelect_neutral = sum(n * z for n, z in zip(atom_counts, zvals))
    nelect_neutral = int(round(nelect_neutral))
    
    # 计算带电体系的电子数
    # 正电荷 = 移除电子，负电荷 = 添加电子
    nelect = nelect_neutral - charge_state
    
    logger.info(f"NELECT calculation: neutral={nelect_neutral}, "
                f"charge={charge_state:+d}, final={nelect}")
    
    # 生成 INCAR 设置
    incar_settings = {
        'NELECT': nelect,
        # 带电缺陷计算通常需要以下设置
        'LMONO': '.TRUE.',  # Monopole correction (optional but recommended)
        'IDIPOL': 4,  # Calculate all dipole moments
        'LDIPOL': '.TRUE.',  # Enable dipole correction
    }
    
    notes = (f"Charged defect calculation: q={charge_state:+d}, "
             f"NELECT={nelect} (neutral: {nelect_neutral})")
    
    return ChargedDefectSettings(
        charge_state=charge_state,
        nelect=nelect,
        nelect_neutral=nelect_neutral,
        incar_settings=incar_settings,
        notes=notes
    )


def get_charged_defect_incar_settings(
    charge_state: int,
    nelect: int,
    include_corrections: bool = True
) -> Dict[str, Any]:
    """
    Get INCAR settings for charged defect calculations.
    
    获取带电缺陷计算所需的 INCAR 参数。
    
    Args:
        charge_state: Charge state of the defect
        nelect: Total number of electrons
        include_corrections: Whether to include monopole/dipole corrections
        
    Returns:
        Dictionary of INCAR parameters
    """
    incar = {
        'NELECT': nelect,
    }
    
    if include_corrections and charge_state != 0:
        # 对于带电体系，添加电荷修正
        incar.update({
            # 偶极矩修正 (对于周期性边界条件下的带电体系很重要)
            'LDIPOL': '.TRUE.',
            'IDIPOL': 4,  # 计算所有方向的偶极矩
            # 可选：单极修正
            # 'LMONO': '.TRUE.',  # 只有在特定 VASP 版本可用
        })
        
        # 建议设置
        logger.info(f"Charged defect (q={charge_state:+d}): "
                    f"Including dipole corrections in INCAR")
    
    return incar


def prepare_charged_defect_calculation(
    task_description: str,
    poscar_path: str,
    potcar_path: str
) -> Optional[ChargedDefectSettings]:
    """
    Prepare settings for charged defect calculation from task description.
    
    从任务描述中准备带电缺陷计算的设置。
    这是主要的入口函数，会：
    1. 解析任务描述中的电荷状态
    2. 计算 NELECT
    3. 返回完整的设置
    
    Args:
        task_description: Natural language task description
        poscar_path: Path to POSCAR file
        potcar_path: Path to POTCAR file
        
    Returns:
        ChargedDefectSettings or None if charge state not found or calculation fails
    """
    # 解析电荷状态
    charge_state = parse_charge_state_from_text(task_description)
    
    if charge_state is None:
        logger.debug("No charge state found in task description")
        return None
    
    # 计算 NELECT 和相关设置
    settings = calculate_nelect(poscar_path, potcar_path, charge_state)
    
    if settings:
        logger.info(f"Prepared charged defect settings: {settings.notes}")
    
    return settings


def extract_defect_formation_energy(
    defect_energy: float,
    bulk_energy: float,
    chemical_potentials: Dict[str, float],
    added_atoms: Dict[str, int],
    removed_atoms: Dict[str, int],
    charge_state: int,
    fermi_level: float,
    vbm_energy: float,
    correction_energy: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate defect formation energy.
    
    计算缺陷形成能。
    
    Formula:
    E_f = E_defect - E_bulk - sum(n_i * mu_i) + q * (E_F + E_VBM) + E_corr
    
    where:
    - E_defect: Total energy of defect supercell
    - E_bulk: Total energy of perfect bulk supercell
    - n_i: Number of atoms added (positive) or removed (negative) of species i
    - mu_i: Chemical potential of species i
    - q: Charge state
    - E_F: Fermi level relative to VBM
    - E_VBM: VBM energy
    - E_corr: Finite-size correction energy
    
    Args:
        defect_energy: Total energy of defect supercell (eV)
        bulk_energy: Total energy of bulk supercell (eV)
        chemical_potentials: Chemical potentials for each species (eV)
        added_atoms: Dict of {element: count} for added atoms
        removed_atoms: Dict of {element: count} for removed atoms
        charge_state: Charge state of the defect
        fermi_level: Fermi level relative to VBM (eV)
        vbm_energy: VBM energy (eV)
        correction_energy: Finite-size correction (eV)
        
    Returns:
        Dictionary with formation energy and breakdown
    """
    # 基础能量差
    delta_e = defect_energy - bulk_energy
    
    # 化学势项
    chemical_term = 0.0
    chemical_breakdown = {}
    
    for element, count in added_atoms.items():
        if element in chemical_potentials:
            contribution = -count * chemical_potentials[element]
            chemical_term += contribution
            chemical_breakdown[f"+{element}"] = contribution
        else:
            logger.warning(f"Missing chemical potential for {element}")
    
    for element, count in removed_atoms.items():
        if element in chemical_potentials:
            contribution = count * chemical_potentials[element]
            chemical_term += contribution
            chemical_breakdown[f"-{element}"] = contribution
        else:
            logger.warning(f"Missing chemical potential for {element}")
    
    # 电荷项
    charge_term = charge_state * (fermi_level + vbm_energy)
    
    # 总形成能
    formation_energy = delta_e + chemical_term + charge_term + correction_energy
    
    result = {
        'formation_energy': formation_energy,
        'defect_energy': defect_energy,
        'bulk_energy': bulk_energy,
        'delta_e': delta_e,
        'chemical_term': chemical_term,
        'chemical_breakdown': chemical_breakdown,
        'charge_term': charge_term,
        'charge_state': charge_state,
        'fermi_level': fermi_level,
        'correction_energy': correction_energy,
    }
    
    logger.info(f"Defect formation energy: {formation_energy:.4f} eV "
                f"(q={charge_state:+d}, E_F={fermi_level:.3f} eV)")
    
    return result
