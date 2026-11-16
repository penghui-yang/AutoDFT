"""
Rule-based recovery handler used as the safety net by ReasoningRecovery.
"""
import os
import re
import shutil
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path

from agents.rule_monitor import FailureReason, SuggestedFix


@dataclass
class RecoveryConfig:
    """
    Configuration for recovery behavior.
    
    Attributes:
        archive_pattern: Pattern for archive directory names
        max_retry_attempts: Maximum number of retry attempts
        preserve_wavecar: Whether to preserve WAVECAR for restart
        preserve_chgcar: Whether to preserve CHGCAR for restart
        restart_from_contcar: Whether to use CONTCAR as new POSCAR
        backup_all_outputs: Whether to backup all output files
    """
    archive_pattern: str = "attempt_{attempt}_{reason}"
    max_retry_attempts: int = 3
    preserve_wavecar: bool = False  # Default: fresh start for stability
    preserve_chgcar: bool = False
    restart_from_contcar: bool = True
    backup_all_outputs: bool = True


@dataclass
class IncarModification:
    """
    A single INCAR parameter modification.
    
    Attributes:
        parameter: INCAR parameter name
        action: 'set', 'increase', 'decrease', 'remove'
        value: New value (for 'set') or factor (for increase/decrease)
        reason: Human-readable reason for this change
    """
    parameter: str
    action: str  # 'set', 'increase', 'decrease', 'remove'
    value: Any
    reason: str = ""


@dataclass
class RecoveryResult:
    """
    Result of a recovery attempt.
    
    Attributes:
        success: Whether recovery setup was successful
        archived_to: Path where failed calculation was archived
        modifications_applied: List of INCAR changes made
        message: Human-readable status message
        attempt_number: Which retry attempt this is
        can_retry: Whether further retries are possible
    """
    success: bool
    archived_to: Optional[str] = None
    modifications_applied: List[IncarModification] = field(default_factory=list)
    message: str = ""
    attempt_number: int = 1
    can_retry: bool = True


class RuleRecoveryHandler:
    """
    Handler for managing failed calculation recovery.
    
    This class provides methods to:
    1. Archive failed calculations safely
    2. Prepare directory for retry
    3. Modify INCAR based on failure diagnosis
    4. Track retry attempts
    
    Attributes:
        calc_dir: Directory containing the failed calculation
        config: Recovery configuration options
    """
    
    # Mapping of suggested fixes to INCAR modifications
    FIX_MODIFICATIONS: Dict[SuggestedFix, List[IncarModification]] = {
        SuggestedFix.SWITCH_ALGO_NORMAL: [
            IncarModification("ALGO", "set", "Normal", "Switched from Fast to Normal for better stability"),
            IncarModification("TIME", "set", 0.4, "Reduced TIME step for safer convergence"),
        ],
        SuggestedFix.SWITCH_ALGO_ALL: [
            IncarModification("ALGO", "set", "All", "Using conjugate gradient + Davidson for difficult cases"),
        ],
        SuggestedFix.REDUCE_POTIM: [
            IncarModification("POTIM", "decrease", 0.5, "Reduced POTIM by 50% for stability"),
        ],
        SuggestedFix.INCREASE_POTIM: [
            IncarModification("POTIM", "increase", 1.5, "Increased POTIM by 50% to escape stagnation"),
        ],
        SuggestedFix.REDUCE_AMIX: [
            IncarModification("AMIX", "set", 0.1, "Reduced AMIX for charge mixing stability"),
            IncarModification("BMIX", "set", 0.001, "Reduced BMIX for charge mixing stability"),
            IncarModification("AMIX_MAG", "set", 0.2, "Reduced magnetic mixing"),
            IncarModification("BMIX_MAG", "set", 0.001, "Reduced magnetic mixing"),
        ],
        SuggestedFix.REDUCE_BMIX: [
            IncarModification("BMIX", "decrease", 0.5, "Reduced BMIX by 50%"),
        ],
        SuggestedFix.INCREASE_NBANDS: [
            IncarModification("NBANDS", "increase", 1.2, "Increased NBANDS by 20%"),
        ],
        SuggestedFix.INCREASE_NELM: [
            IncarModification("NELM", "set", 200, "Increased NELM limit"),
        ],
        SuggestedFix.INCREASE_EDIFF: [
            IncarModification("EDIFF", "increase", 10.0, "Relaxed EDIFF by 10x for initial convergence"),
        ],
        SuggestedFix.SET_ADDGRID: [
            IncarModification("ADDGRID", "set", ".TRUE.", "Added fine FFT grid"),
        ],
        SuggestedFix.INCREASE_ENCUT: [
            IncarModification("ENCUT", "increase", 1.1, "Increased ENCUT by 10%"),
        ],
        SuggestedFix.SET_LREAL_FALSE: [
            IncarModification("LREAL", "set", ".FALSE.", "Using reciprocal space projection"),
        ],
        SuggestedFix.SET_ISMEAR_0: [
            IncarModification("ISMEAR", "set", 0, "Using Gaussian smearing"),
            IncarModification("SIGMA", "set", 0.05, "Small smearing width"),
        ],
        SuggestedFix.REDUCE_SYMPREC: [
            IncarModification("SYMPREC", "set", 1e-6, "Reduced symmetry precision"),
        ],
        SuggestedFix.INCREASE_SYMPREC: [
            IncarModification("SYMPREC", "set", 1e-4, "Increased symmetry precision"),
        ],
        SuggestedFix.SET_ISYM_0: [
            IncarModification("ISYM", "set", 0, "Disabled symmetry"),
        ],
        SuggestedFix.FRESH_START: [
            # No INCAR changes, just fresh start without WAVECAR/CHGCAR
            IncarModification("ISTART", "set", 0, "Fresh start from scratch"),
        ],
        SuggestedFix.SCALE_STRUCTURE: [
            # This requires POSCAR modification, handled specially
        ],
    }
    
    def __init__(
        self,
        calc_dir: str,
        config: Optional[RecoveryConfig] = None
    ):
        """
        Initialize the recovery handler.
        
        Args:
            calc_dir: Directory containing the calculation to recover
            config: Optional recovery configuration (uses defaults if None)
        """
        self.calc_dir = os.path.abspath(calc_dir)
        self.config = config or RecoveryConfig()
        self._attempt_number = self._count_previous_attempts()
        
        logging.info(f"RuleRecoveryHandler initialized for {calc_dir}")
        logging.info(f"  Previous attempts: {self._attempt_number}")
    
    def _count_previous_attempts(self) -> int:
        """
        Count previous retry attempts by looking for archive directories.
        
        Returns:
            Number of previous attempts (0 if none)
        """
        if not os.path.exists(self.calc_dir):
            return 0
        
        attempt_count = 0
        for item in os.listdir(self.calc_dir):
            if item.startswith("attempt_") and os.path.isdir(os.path.join(self.calc_dir, item)):
                attempt_count += 1
        
        return attempt_count
    
    def can_retry(self) -> bool:
        """
        Check if further retries are possible.
        
        Returns:
            True if retry attempts remain, False otherwise
        """
        return self._attempt_number < self.config.max_retry_attempts
    
    def get_attempt_number(self) -> int:
        """
        Get the current attempt number (1-indexed).
        
        Returns:
            Next attempt number
        """
        return self._attempt_number + 1
    
    def prepare_retry(
        self,
        failure_reason: FailureReason,
        suggested_fix: SuggestedFix,
        custom_modifications: Optional[List[IncarModification]] = None
    ) -> RecoveryResult:
        """
        Prepare the calculation directory for a retry.
        
        This is the main entry point for recovery. It:
        1. Archives the failed calculation
        2. Restores input files
        3. Applies INCAR modifications
        
        Args:
            failure_reason: Diagnosed failure reason from RuleMonitor
            suggested_fix: Suggested fix from RuleMonitor
            custom_modifications: Optional additional INCAR modifications
            
        Returns:
            RecoveryResult with status and details
        """
        logging.info(f"Preparing retry for {self.calc_dir}")
        logging.info(f"  Failure reason: {failure_reason.value}")
        logging.info(f"  Suggested fix: {suggested_fix.value}")
        
        # Check if we can retry
        if not self.can_retry():
            return RecoveryResult(
                success=False,
                message=f"Maximum retry attempts ({self.config.max_retry_attempts}) reached",
                attempt_number=self._attempt_number,
                can_retry=False
            )
        
        try:
            # Step 1: Archive the failed calculation
            archive_path = self._archive_failed_calculation(failure_reason)
            
            # Step 2: Restore input files
            self._restore_input_files(archive_path, suggested_fix)
            
            # Step 3: Apply INCAR modifications
            modifications = self._get_modifications(suggested_fix, custom_modifications)
            self._apply_incar_modifications(modifications)
            
            # Step 4: Handle special cases (like structure scaling)
            if suggested_fix == SuggestedFix.SCALE_STRUCTURE:
                self._scale_structure()
            
            self._attempt_number += 1
            
            return RecoveryResult(
                success=True,
                archived_to=archive_path,
                modifications_applied=modifications,
                message=f"Successfully prepared retry attempt {self._attempt_number}",
                attempt_number=self._attempt_number,
                can_retry=self._attempt_number < self.config.max_retry_attempts
            )
            
        except Exception as e:
            logging.error(f"Recovery failed: {e}")
            return RecoveryResult(
                success=False,
                message=f"Recovery failed: {str(e)}",
                attempt_number=self._attempt_number,
                can_retry=True  # Still can retry if setup failed
            )
    
    def _archive_failed_calculation(self, failure_reason: FailureReason) -> str:
        """
        Archive the failed calculation to a subdirectory.
        
        Args:
            failure_reason: Reason for the failure
            
        Returns:
            Path to the archive directory
        """
        attempt_num = self._attempt_number + 1
        reason_str = failure_reason.value.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        archive_name = f"attempt_{attempt_num}_{reason_str}_{timestamp}"
        archive_path = os.path.join(self.calc_dir, archive_name)
        
        os.makedirs(archive_path, exist_ok=True)
        logging.info(f"Archiving failed calculation to {archive_path}")
        
        # Files to archive (VASP outputs)
        output_files = [
            "OUTCAR", "OSZICAR", "CONTCAR", "CHGCAR", "WAVECAR",
            "vasprun.xml", "DOSCAR", "EIGENVAL", "PROCAR", "PCDAT",
            "XDATCAR", "IBZKPT", "CHG", "LOCPOT", "ELFCAR",
            "vasp_run.out", "vasp_run.err", "print-out",
            "submit_vasp.sh", "REPORT"
        ]
        
        # Also archive input files for reference
        input_files = ["INCAR", "POSCAR", "KPOINTS", "POTCAR"]
        
        all_files = output_files + input_files if self.config.backup_all_outputs else output_files
        
        for filename in all_files:
            src_path = os.path.join(self.calc_dir, filename)
            if os.path.exists(src_path):
                dst_path = os.path.join(archive_path, filename)
                if os.path.isfile(src_path):
                    shutil.move(src_path, dst_path)
                    logging.debug(f"  Archived {filename}")
        
        # Write a summary file
        summary_path = os.path.join(archive_path, "RECOVERY_INFO.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Archived: {timestamp}\n")
            f.write(f"Attempt: {attempt_num}\n")
            f.write(f"Failure Reason: {failure_reason.value}\n")
            f.write(f"Directory: {self.calc_dir}\n")
        
        return archive_path
    
    def _restore_input_files(self, archive_path: str, suggested_fix: SuggestedFix):
        """
        Restore input files for the retry.
        
        This handles the decision of whether to use:
        - Original POSCAR or CONTCAR from failed calculation
        - Fresh start vs continuing with WAVECAR/CHGCAR
        
        Args:
            archive_path: Path to the archived failed calculation
            suggested_fix: The suggested fix (affects restart strategy)
        """
        logging.info("Restoring input files for retry")
        
        # Determine restart strategy based on failure type
        use_wavecar = self.config.preserve_wavecar
        use_chgcar = self.config.preserve_chgcar
        use_contcar = self.config.restart_from_contcar
        
        # Fresh start is safer for divergence issues
        if suggested_fix in [SuggestedFix.FRESH_START, SuggestedFix.SWITCH_ALGO_NORMAL,
                            SuggestedFix.SWITCH_ALGO_ALL, SuggestedFix.REDUCE_AMIX]:
            use_wavecar = False
            use_chgcar = False
            logging.info("  Using fresh start (no WAVECAR/CHGCAR)")
        
        # Structure issues need original POSCAR
        if suggested_fix in [SuggestedFix.SCALE_STRUCTURE, SuggestedFix.USE_DIFFERENT_POSCAR]:
            use_contcar = False
            logging.info("  Using original POSCAR (structure issue)")
        
        # Restore POSCAR
        self._restore_poscar(archive_path, use_contcar)
        
        # Restore POTCAR (always from archive)
        potcar_src = os.path.join(archive_path, "POTCAR")
        if os.path.exists(potcar_src):
            shutil.copy(potcar_src, os.path.join(self.calc_dir, "POTCAR"))
            logging.info("  Restored POTCAR")
        
        # Restore KPOINTS (always from archive)
        kpoints_src = os.path.join(archive_path, "KPOINTS")
        if os.path.exists(kpoints_src):
            shutil.copy(kpoints_src, os.path.join(self.calc_dir, "KPOINTS"))
            logging.info("  Restored KPOINTS")
        
        # Restore INCAR (will be modified later)
        incar_src = os.path.join(archive_path, "INCAR")
        if os.path.exists(incar_src):
            shutil.copy(incar_src, os.path.join(self.calc_dir, "INCAR"))
            logging.info("  Restored INCAR (will be modified)")
        
        # Optionally restore WAVECAR/CHGCAR for continuation
        if use_wavecar:
            wavecar_src = os.path.join(archive_path, "WAVECAR")
            if os.path.exists(wavecar_src):
                shutil.copy(wavecar_src, os.path.join(self.calc_dir, "WAVECAR"))
                logging.info("  Restored WAVECAR for continuation")
        
        if use_chgcar:
            chgcar_src = os.path.join(archive_path, "CHGCAR")
            if os.path.exists(chgcar_src):
                shutil.copy(chgcar_src, os.path.join(self.calc_dir, "CHGCAR"))
                logging.info("  Restored CHGCAR for continuation")
    
    def _restore_poscar(self, archive_path: str, use_contcar: bool):
        """
        Restore POSCAR file with appropriate strategy.
        
        Args:
            archive_path: Path to archive directory
            use_contcar: Whether to use CONTCAR as new POSCAR
        """
        if use_contcar:
            # Try to use CONTCAR if it exists and is valid
            contcar_src = os.path.join(archive_path, "CONTCAR")
            if os.path.exists(contcar_src) and os.path.getsize(contcar_src) > 0:
                # Validate CONTCAR has atomic positions
                if self._is_valid_poscar(contcar_src):
                    shutil.copy(contcar_src, os.path.join(self.calc_dir, "POSCAR"))
                    logging.info("  Restored CONTCAR as POSCAR (structure continuation)")
                    return
                else:
                    logging.warning("  CONTCAR exists but appears invalid, using original POSCAR")
        
        # Fallback to original POSCAR
        poscar_src = os.path.join(archive_path, "POSCAR")
        if os.path.exists(poscar_src):
            shutil.copy(poscar_src, os.path.join(self.calc_dir, "POSCAR"))
            logging.info("  Restored original POSCAR")
        else:
            raise FileNotFoundError("No POSCAR found in archive")
    
    def _is_valid_poscar(self, poscar_path: str) -> bool:
        """
        Check if a POSCAR file is valid and has atomic positions.
        
        Args:
            poscar_path: Path to POSCAR file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 8:
                return False
            
            # Check for atom counts line (line 6, 0-indexed = 5)
            counts_line = lines[6].strip()
            counts = [int(x) for x in counts_line.split()]
            total_atoms = sum(counts)
            
            if total_atoms <= 0:
                return False
            
            # Check that we have enough position lines
            # Position lines start after line 7 (or 8 if selective dynamics)
            start_line = 7
            if lines[7].strip()[0].upper() == 'S':
                start_line = 8
            
            # Skip coordinate type line
            start_line += 1
            
            # Count position lines
            position_lines = len(lines) - start_line
            if position_lines < total_atoms:
                return False
            
            return True
            
        except Exception as e:
            logging.debug(f"POSCAR validation failed: {e}")
            return False
    
    def _get_modifications(
        self,
        suggested_fix: SuggestedFix,
        custom_modifications: Optional[List[IncarModification]] = None
    ) -> List[IncarModification]:
        """
        Get the list of INCAR modifications to apply.
        
        Args:
            suggested_fix: Suggested fix from monitoring
            custom_modifications: Additional custom modifications
            
        Returns:
            List of modifications to apply
        """
        modifications = []
        
        # Get standard modifications for this fix
        if suggested_fix in self.FIX_MODIFICATIONS:
            modifications.extend(self.FIX_MODIFICATIONS[suggested_fix])
        
        # Add custom modifications
        if custom_modifications:
            modifications.extend(custom_modifications)
        
        return modifications
    
    def _apply_incar_modifications(self, modifications: List[IncarModification]):
        """
        Apply modifications to the INCAR file.
        
        Args:
            modifications: List of IncarModification objects to apply
        """
        incar_path = os.path.join(self.calc_dir, "INCAR")
        
        if not os.path.exists(incar_path):
            logging.warning("No INCAR found, creating new one")
            with open(incar_path, 'w') as f:
                f.write("# INCAR created by RuleRecoveryHandler\n")
        
        # Read current INCAR
        incar_params = self._parse_incar(incar_path)
        
        # Apply modifications
        for mod in modifications:
            old_value = incar_params.get(mod.parameter)
            
            if mod.action == "set":
                incar_params[mod.parameter] = mod.value
                logging.info(f"  Set {mod.parameter} = {mod.value} (was: {old_value})")
                
            elif mod.action == "increase":
                if old_value is not None:
                    try:
                        new_value = float(old_value) * mod.value
                        # Keep as int if original was int-like
                        if isinstance(old_value, int) or (isinstance(old_value, float) and old_value.is_integer()):
                            new_value = int(new_value)
                        incar_params[mod.parameter] = new_value
                        logging.info(f"  Increased {mod.parameter}: {old_value} -> {new_value}")
                    except (ValueError, TypeError):
                        logging.warning(f"  Could not increase {mod.parameter}, setting to {mod.value}")
                        incar_params[mod.parameter] = mod.value
                else:
                    logging.debug(f"  {mod.parameter} not found, skipping increase")
                    
            elif mod.action == "decrease":
                if old_value is not None:
                    try:
                        new_value = float(old_value) * mod.value
                        if isinstance(old_value, int) or (isinstance(old_value, float) and old_value.is_integer()):
                            new_value = int(new_value)
                        incar_params[mod.parameter] = new_value
                        logging.info(f"  Decreased {mod.parameter}: {old_value} -> {new_value}")
                    except (ValueError, TypeError):
                        logging.warning(f"  Could not decrease {mod.parameter}")
                        
            elif mod.action == "remove":
                if mod.parameter in incar_params:
                    del incar_params[mod.parameter]
                    logging.info(f"  Removed {mod.parameter} (was: {old_value})")
        
        # Write modified INCAR
        self._write_incar(incar_path, incar_params, modifications)
    
    def _parse_incar(self, incar_path: str) -> Dict[str, Any]:
        """
        Parse INCAR file into a dictionary.
        
        Args:
            incar_path: Path to INCAR file
            
        Returns:
            Dictionary of parameter: value pairs
        """
        params = {}
        
        try:
            with open(incar_path, 'r') as f:
                for line in f:
                    # Skip comments and empty lines
                    line = line.split('#')[0].split('!')[0].strip()
                    if not line or '=' not in line:
                        continue
                    
                    # Parse parameter = value
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().upper()
                        value = parts[1].strip()
                        
                        # Try to convert to appropriate type
                        params[key] = self._parse_incar_value(value)
        
        except Exception as e:
            logging.warning(f"Error parsing INCAR: {e}")
        
        return params
    
    def _parse_incar_value(self, value_str: str) -> Any:
        """
        Parse an INCAR value string to appropriate Python type.
        
        Args:
            value_str: Value string from INCAR
            
        Returns:
            Parsed value (int, float, bool, or string)
        """
        value_str = value_str.strip()
        
        # Boolean values
        if value_str.upper() in ['.TRUE.', 'T', 'TRUE', '.T.']:
            return '.TRUE.'
        if value_str.upper() in ['.FALSE.', 'F', 'FALSE', '.F.']:
            return '.FALSE.'
        
        # Try integer
        try:
            return int(value_str)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # Return as string
        return value_str
    
    def _write_incar(
        self,
        incar_path: str,
        params: Dict[str, Any],
        modifications: List[IncarModification]
    ):
        """
        Write INCAR file with parameters.
        
        Args:
            incar_path: Path to write INCAR
            params: Dictionary of parameters
            modifications: List of modifications applied (for comment)
        """
        with open(incar_path, 'w') as f:
            f.write("# INCAR modified by RuleRecoveryHandler\n")
            f.write(f"# Modified: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Attempt: {self._attempt_number + 1}\n")
            
            # Write modification summary
            if modifications:
                f.write("# Modifications applied:\n")
                for mod in modifications:
                    f.write(f"#   {mod.parameter}: {mod.action} -> {mod.value}\n")
                    if mod.reason:
                        f.write(f"#     Reason: {mod.reason}\n")
            
            f.write("\n")
            
            # Group parameters by category (for readability)
            categories = {
                "General": ["SYSTEM", "ISTART", "ICHARG", "PREC", "ENCUT", "LREAL"],
                "Electronic": ["EDIFF", "NELM", "NELMIN", "ALGO", "AMIX", "BMIX", 
                              "AMIX_MAG", "BMIX_MAG", "TIME", "IALGO"],
                "Ionic": ["IBRION", "NSW", "ISIF", "EDIFFG", "POTIM"],
                "Smearing": ["ISMEAR", "SIGMA"],
                "Symmetry": ["ISYM", "SYMPREC"],
                "Spin": ["ISPIN", "MAGMOM", "LORBIT"],
                "Bands": ["NBANDS", "NEDOS"],
                "Output": ["LWAVE", "LCHARG", "LAECHG", "LVTOT", "LVHAR"],
            }
            
            written = set()
            
            for category, cat_params in categories.items():
                cat_entries = []
                for p in cat_params:
                    if p in params:
                        cat_entries.append((p, params[p]))
                        written.add(p)
                
                if cat_entries:
                    f.write(f"# {category}\n")
                    for p, v in cat_entries:
                        f.write(f"{p} = {v}\n")
                    f.write("\n")
            
            # Write remaining parameters
            remaining = [(k, v) for k, v in params.items() if k not in written]
            if remaining:
                f.write("# Other\n")
                for k, v in remaining:
                    f.write(f"{k} = {v}\n")
        
        logging.info(f"  Written modified INCAR with {len(params)} parameters")
    
    def _scale_structure(self, scale_factor: float = 1.02):
        """
        Scale the structure (increase lattice parameters) for atom overlap issues.
        
        Args:
            scale_factor: Factor to scale lattice vectors (default 2% increase)
        """
        poscar_path = os.path.join(self.calc_dir, "POSCAR")
        
        if not os.path.exists(poscar_path):
            logging.warning("Cannot scale structure: POSCAR not found")
            return
        
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            # Modify scaling factor (line 1, 0-indexed)
            old_scale = float(lines[1].strip())
            new_scale = old_scale * scale_factor
            lines[1] = f"  {new_scale:.10f}\n"
            
            with open(poscar_path, 'w') as f:
                f.writelines(lines)
            
            logging.info(f"  Scaled structure: {old_scale:.6f} -> {new_scale:.6f} "
                        f"(factor: {scale_factor})")
            
        except Exception as e:
            logging.error(f"Failed to scale structure: {e}")
    
    def get_archive_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all archived attempts.
        
        Returns:
            List of dictionaries with archive info
        """
        history = []
        
        if not os.path.exists(self.calc_dir):
            return history
        
        for item in sorted(os.listdir(self.calc_dir)):
            if item.startswith("attempt_") and os.path.isdir(os.path.join(self.calc_dir, item)):
                archive_path = os.path.join(self.calc_dir, item)
                info_file = os.path.join(archive_path, "RECOVERY_INFO.txt")
                
                info = {
                    "name": item,
                    "path": archive_path,
                }
                
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        for line in f:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                info[key.strip().lower()] = value.strip()
                
                history.append(info)
        
        return history


def create_rule_recovery_handler(
    calc_dir: str,
    **config_kwargs
) -> RuleRecoveryHandler:
    """
    Factory function to create a RuleRecoveryHandler with optional configuration.
    
    Args:
        calc_dir: Directory containing the calculation
        **config_kwargs: Configuration parameters for RecoveryConfig
        
    Returns:
        Configured RuleRecoveryHandler instance
    """
    config = RecoveryConfig(**config_kwargs) if config_kwargs else None
    return RuleRecoveryHandler(calc_dir, config)
