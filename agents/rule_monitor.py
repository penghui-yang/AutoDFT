"""
Rule-based monitor for running VASP calculations.
"""
import os
import re
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime


class CalculationStatus(Enum):
    """Status codes for calculation assessment."""
    CONTINUE = "CONTINUE"      # Calculation proceeding normally
    KILL = "KILL"              # Calculation should be terminated
    CONVERGED = "CONVERGED"    # Calculation finished successfully
    NOT_STARTED = "NOT_STARTED"  # No output files found yet


class FailureReason(Enum):
    """Specific failure/issue codes for diagnosis."""
    NONE = "NONE"
    ELECTRONIC_DIVERGENCE = "ELECTRONIC_DIVERGENCE"
    ENERGY_OSCILLATION = "ENERGY_OSCILLATION"
    FORCE_STAGNATION = "FORCE_STAGNATION"
    POSITIVE_ENERGY = "POSITIVE_ENERGY"
    EXTREME_FORCES = "EXTREME_FORCES"
    ATOM_OVERLAP = "ATOM_OVERLAP"
    GRID_ERROR = "GRID_ERROR"
    ZBRENT_ERROR = "ZBRENT_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    EDDDAV_ERROR = "EDDDAV_ERROR"
    BRMIX_ERROR = "BRMIX_ERROR"
    PRICEL_ERROR = "PRICEL_ERROR"
    SYMPREC_ERROR = "SYMPREC_ERROR"
    CHGCAR_READ_ERROR = "CHGCAR_READ_ERROR"
    MAGMOM_ERROR = "MAGMOM_ERROR"
    HYBRID_ICHARG_CONFLICT = "HYBRID_ICHARG_CONFLICT"
    ADDGRID_TS_CONFLICT = "ADDGRID_TS_CONFLICT"
    KPOINTS_MODE_CONFLICT = "KPOINTS_MODE_CONFLICT"
    NOMEGA_LIMIT = "NOMEGA_LIMIT"
    INSUFFICIENT_BANDS = "INSUFFICIENT_BANDS"
    WAVECAR_BASIS_CHANGE = "WAVECAR_BASIS_CHANGE"
    EDDDAV_SUBSPACE_ERROR = "EDDDAV_SUBSPACE_ERROR"
    TIMEOUT = "TIMEOUT"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class SuggestedFix(Enum):
    """Suggested fixes mapped to failure reasons."""
    NONE = "NONE"
    INCREASE_NBANDS = "INCREASE_NBANDS"
    SWITCH_ALGO_NORMAL = "SWITCH_ALGO_NORMAL"
    SWITCH_ALGO_ALL = "SWITCH_ALGO_ALL"
    REDUCE_POTIM = "REDUCE_POTIM"
    INCREASE_POTIM = "INCREASE_POTIM"
    REDUCE_AMIX = "REDUCE_AMIX"
    REDUCE_BMIX = "REDUCE_BMIX"
    INCREASE_NELM = "INCREASE_NELM"
    INCREASE_EDIFF = "INCREASE_EDIFF"
    SET_ADDGRID = "SET_ADDGRID"
    INCREASE_ENCUT = "INCREASE_ENCUT"
    SET_LREAL_FALSE = "SET_LREAL_FALSE"
    SET_ISMEAR_0 = "SET_ISMEAR_0"
    REDUCE_SYMPREC = "REDUCE_SYMPREC"
    INCREASE_SYMPREC = "INCREASE_SYMPREC"
    SET_ISYM_0 = "SET_ISYM_0"
    FRESH_START = "FRESH_START"
    SCALE_STRUCTURE = "SCALE_STRUCTURE"
    USE_DIFFERENT_POSCAR = "USE_DIFFERENT_POSCAR"
    INCREASE_MEMORY = "INCREASE_MEMORY"
    FIX_MAGMOM = "FIX_MAGMOM"
    FIX_HYBRID_RESTART = "FIX_HYBRID_RESTART"
    DISABLE_ADDGRID = "DISABLE_ADDGRID"
    FIX_KPOINTS_MESH = "FIX_KPOINTS_MESH"
    REDUCE_NOMEGA = "REDUCE_NOMEGA"


@dataclass
class IonicStepData:
    """Data from a single ionic step (from OSZICAR)."""
    step_number: int
    n_electronic_steps: int
    E0: float  # Total energy without entropy
    dE: float  # Energy change from previous step
    mag: Optional[float] = None  # Magnetic moment (if ISPIN=2)


@dataclass
class AssessmentResult:
    """
    Structured result from calculation assessment.
    
    Attributes:
        status: Overall assessment (CONTINUE, KILL, CONVERGED)
        failure_reason: Specific error code if status is KILL
        suggested_fix: Hint for the recovery handler
        details: Human-readable description of the issue
        metrics: Quantitative data supporting the diagnosis
        total_elapsed_time: Time since calculation started (seconds)
        time_since_last_check: Time since last monitoring check (seconds)
        ionic_steps_completed: Number of ionic relaxation steps done
        electronic_convergence_history: List of electronic step counts per ionic step
        reset_timeout: Whether to reset the timeout counter (e.g., after significant progress)
        milestone_reached: Description of the milestone that triggered reset (if any)
    """
    status: CalculationStatus
    failure_reason: FailureReason = FailureReason.NONE
    suggested_fix: SuggestedFix = SuggestedFix.NONE
    details: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    total_elapsed_time: float = 0.0
    time_since_last_check: float = 0.0
    ionic_steps_completed: int = 0
    electronic_convergence_history: List[int] = field(default_factory=list)
    reset_timeout: bool = False  # Signal to job_manager to reset timeout counter
    milestone_reached: str = ""  # Description of milestone (e.g., "SOC SAXIS 001 completed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "status": self.status.value,
            "failure_reason": self.failure_reason.value,
            "suggested_fix": self.suggested_fix.value,
            "details": self.details,
            "metrics": self.metrics,
            "total_elapsed_time": self.total_elapsed_time,
            "time_since_last_check": self.time_since_last_check,
            "ionic_steps_completed": self.ionic_steps_completed,
            "electronic_convergence_history": self.electronic_convergence_history,
            "reset_timeout": self.reset_timeout,
            "milestone_reached": self.milestone_reached,
        }


class RuleMonitor:
    """
    Agent for monitoring running VASP calculations.
    
    This agent periodically inspects OSZICAR and OUTCAR files to:
    1. Detect electronic divergence (hitting NELM limit)
    2. Detect energy oscillation (no downward trend)
    3. Detect stagnation (small dE but forces not converging)
    4. Detect abnormal physics (positive energies, extreme forces)
    5. Detect VASP-specific errors from OUTCAR
    6. Handle DFPT calculations (IBRION=7,8) which don't have ionic steps
    7. Handle SOC calculations (LSORBIT=.TRUE.) with SAXIS progress tracking
    
    For DFPT/Raman/phonon calculations (IBRION=7 or 8), the agent recognizes
    that these calculations don't produce standard ionic step output and
    reports progress based on perturbations completed instead.
    
    For SOC calculations, the agent tracks:
    - Electronic convergence progress
    - SAXIS direction changes (important milestones)
    - Can signal timeout reset when significant progress is made
    
    Attributes:
        nelm_threshold: Number of electronic steps indicating NELM hit (default 60)
        oscillation_window: Number of ionic steps to check for oscillation (default 10)
        stagnation_de_threshold: dE threshold for stagnation detection (eV)
        force_threshold: Force threshold for abnormal physics (eV/Å)
        energy_threshold: Positive energy threshold (eV)
        check_interval: Recommended check interval in seconds (default 600 = 10 min)
        _last_soc_progress: Internal state for tracking SOC progress milestones
    """
    
    def __init__(
        self,
        nelm_threshold: int = 60,
        oscillation_window: int = 10,
        stagnation_de_threshold: float = 1e-4,
        stagnation_step_threshold: int = 20,
        force_threshold: float = 100.0,
        energy_threshold: float = 0.0,
        check_interval: int = 600,
    ):
        """
        Initialize the monitoring agent.
        
        Args:
            nelm_threshold: NELM limit to detect electronic divergence
            oscillation_window: Window size for oscillation detection
            stagnation_de_threshold: dE below which we check for stagnation
            stagnation_step_threshold: Steps with small dE before flagging stagnation
            force_threshold: Force magnitude indicating atom overlap/abnormal physics
            energy_threshold: Energy above which is considered abnormal (default 0)
            check_interval: Recommended interval between checks (seconds)
        """
        self.nelm_threshold = nelm_threshold
        self.oscillation_window = oscillation_window
        self.stagnation_de_threshold = stagnation_de_threshold
        self.stagnation_step_threshold = stagnation_step_threshold
        self.force_threshold = force_threshold
        self.energy_threshold = energy_threshold
        self.check_interval = check_interval
        
        # Internal state for tracking SOC progress milestones
        self._last_soc_progress: Dict[str, Any] = {
            "electronic_steps": 0,
            "scf_converged": False,
            "last_energy": None,
            "milestone_step": 0,  # Track which milestone step we've reached
        }
        
        # SOC milestone configuration: reset timeout every N electronic steps
        # This is important because SOC calculations can be very slow
        # Using a smaller interval (10) because each SOC electronic step can take minutes
        self.soc_milestone_interval = 10  # Reset timeout every 10 electronic steps
        
        logging.info(f"RuleMonitor initialized with NELM threshold={nelm_threshold}, "
                     f"oscillation window={oscillation_window}")
    
    def assess_calculation(
        self,
        calc_dir: str,
        start_time: float,
        last_check_time: Optional[float] = None
    ) -> AssessmentResult:
        """
        Assess the current state of a running VASP calculation.
        
        This is the main entry point for monitoring. It reads OSZICAR and OUTCAR
        to diagnose the calculation state.
        
        Args:
            calc_dir: Directory containing the VASP calculation
            start_time: Unix timestamp when the calculation started
            last_check_time: Unix timestamp of the last monitoring check (or None)
            
        Returns:
            AssessmentResult with status, failure_reason, and suggested_fix
        """
        current_time = time.time()
        total_elapsed = current_time - start_time
        time_since_check = current_time - (last_check_time or start_time)
        
        logging.info(f"Assessing calculation in {calc_dir}")
        logging.info(f"  Total elapsed: {total_elapsed/60:.1f} min, "
                     f"Since last check: {time_since_check/60:.1f} min")
        
        # Initialize result with timing info
        result = AssessmentResult(
            status=CalculationStatus.NOT_STARTED,
            total_elapsed_time=total_elapsed,
            time_since_last_check=time_since_check
        )
        
        oszicar_path = os.path.join(calc_dir, "OSZICAR")
        outcar_path = os.path.join(calc_dir, "OUTCAR")
        
        # Check if files exist
        if not os.path.exists(oszicar_path) and not os.path.exists(outcar_path):
            result.details = "No output files found yet"
            return result
        
        # Step 1: Check OUTCAR for explicit VASP errors
        if os.path.exists(outcar_path):
            outcar_result = self._check_outcar_errors(outcar_path)
            if outcar_result.status == CalculationStatus.KILL:
                outcar_result.total_elapsed_time = total_elapsed
                outcar_result.time_since_last_check = time_since_check
                return outcar_result
            
            # Check for successful completion
            if self._check_convergence_outcar(outcar_path):
                result.status = CalculationStatus.CONVERGED
                result.details = "Calculation converged successfully"
                return result
            
            # Step 1b: Special handling for DFPT calculations (IBRION=7,8)
            # DFPT calculations don't have ionic steps - they work through perturbations
            if self._is_dfpt_calculation(outcar_path):
                dfpt_progress = self._check_dfpt_progress(outcar_path)
                result.status = CalculationStatus.CONTINUE
                
                # Build informative progress message
                if dfpt_progress['total_perturbations'] > 0:
                    progress_str = (
                        f"{dfpt_progress['perturbations_done']}/{dfpt_progress['total_perturbations']} "
                        f"degrees of freedom"
                    )
                elif dfpt_progress['perturbations_done'] > 0:
                    progress_str = f"{dfpt_progress['perturbations_done']} perturbations done"
                elif dfpt_progress['directions_done'] > 0:
                    progress_str = f"{dfpt_progress['directions_done']} directions done (linear response phase)"
                else:
                    progress_str = "initializing"
                
                result.details = (
                    f"DFPT calculation in progress: {progress_str}, "
                    f"current electronic step: {dfpt_progress['electronic_steps_current']}"
                )
                result.metrics = {
                    "calculation_type": "DFPT",
                    "perturbations_done": dfpt_progress["perturbations_done"],
                    "total_perturbations": dfpt_progress["total_perturbations"],
                    "directions_done": dfpt_progress["directions_done"],
                    "electronic_steps_current": dfpt_progress["electronic_steps_current"],
                    "last_energy": dfpt_progress["last_energy"],
                }

                # Reset timeout on DFPT progress milestones (each new perturbation
                # or direction proves the calculation is still making progress).
                prev_done = getattr(self, '_last_dfpt_perturbations_done', 0)
                current_done = dfpt_progress["perturbations_done"] + dfpt_progress["directions_done"]
                if current_done > prev_done:
                    result.reset_timeout = True
                    result.milestone_reached = (
                        f"DFPT progress: {progress_str}"
                    )
                    self._last_dfpt_perturbations_done = current_done

                logging.info(f"DFPT calculation detected: {result.details}")
                return result
            
            # Step 1c: Special handling for SOC calculations (LSORBIT=.TRUE.)
            # SOC calculations are typically static (NSW=0) but need special monitoring
            if self._is_soc_calculation(outcar_path):
                soc_progress = self._check_soc_progress(outcar_path, calc_dir)
                result.status = CalculationStatus.CONTINUE
                
                # Build informative progress message
                nelm = soc_progress.get('nelm', 60)
                elec_step = soc_progress['electronic_step']
                last_de = soc_progress.get('last_de')
                saxis = soc_progress.get('saxis', 'unknown')
                
                if last_de is not None:
                    progress_str = f"electronic step {elec_step}/{nelm}, dE={last_de:.2e}, SAXIS={saxis}"
                else:
                    progress_str = f"electronic step {elec_step}/{nelm}, SAXIS={saxis}"
                
                result.details = f"SOC calculation in progress: {progress_str}"
                result.metrics = {
                    "calculation_type": "SOC",
                    "electronic_step": elec_step,
                    "nelm": nelm,
                    "last_energy": soc_progress.get("last_energy"),
                    "last_de": last_de,
                    "saxis": saxis,
                    "scf_converged": soc_progress.get("scf_converged", False),
                }
                
                # Check for milestones that should reset timeout
                # 1. SCF convergence (most significant milestone)
                if soc_progress.get("scf_converged") and not self._last_soc_progress.get("scf_converged"):
                    result.reset_timeout = True
                    result.milestone_reached = f"SOC SCF converged (SAXIS={saxis})"
                    logging.info(f"[Monitor] SOC milestone reached: {result.milestone_reached}")
                
                # 2. Progress milestone: every N electronic steps (indicates calc is progressing)
                elif elec_step > 0:
                    current_milestone = elec_step // self.soc_milestone_interval
                    last_milestone = self._last_soc_progress.get("milestone_step", 0)
                    
                    if current_milestone > last_milestone:
                        result.reset_timeout = True
                        result.milestone_reached = f"SOC electronic step {elec_step} (SAXIS={saxis})"
                        logging.info(f"[Monitor] SOC progress milestone: {elec_step} electronic steps completed")
                        self._last_soc_progress["milestone_step"] = current_milestone
                
                # Update internal state for tracking milestones
                self._last_soc_progress = {
                    "electronic_steps": elec_step,
                    "scf_converged": soc_progress.get("scf_converged", False),
                    "last_energy": soc_progress.get("last_energy"),
                    "milestone_step": self._last_soc_progress.get("milestone_step", 0),
                }
                
                logging.info(f"SOC calculation detected: {result.details}")
                return result
            
            # Step 1d: Special handling for static calculations (NSW=0, IBRION=-1)
            # Static calculations (SCF, band structure, DOS) don't have ionic steps
            if self._is_static_calculation(outcar_path):
                static_progress = self._check_static_progress(outcar_path)
                result.status = CalculationStatus.CONTINUE
                
                # Build informative progress message
                nelm = static_progress.get('nelm', 60)
                elec_step = static_progress['electronic_step']
                last_de = static_progress.get('last_de')
                
                if last_de is not None:
                    progress_str = f"electronic step {elec_step}/{nelm}, dE={last_de:.2e}"
                else:
                    progress_str = f"electronic step {elec_step}/{nelm}"
                
                result.details = f"Static calculation in progress: {progress_str}"
                result.metrics = {
                    "calculation_type": "static",
                    "electronic_step": elec_step,
                    "nelm": nelm,
                    "last_energy": static_progress.get("last_energy"),
                    "last_de": last_de,
                }

                # Reset timeout periodically for static calculations making progress.
                # LEPSILON / dielectric response calcs can be very slow.
                STATIC_MILESTONE_INTERVAL = 20
                prev_milestone = getattr(self, '_last_static_milestone', 0)
                current_milestone = elec_step // STATIC_MILESTONE_INTERVAL
                if current_milestone > prev_milestone:
                    result.reset_timeout = True
                    result.milestone_reached = f"Static electronic step {elec_step}"
                    self._last_static_milestone = current_milestone

                logging.info(f"Static calculation detected: {result.details}")
                return result
        
        # Step 2: Parse OSZICAR for ionic/electronic step data
        if os.path.exists(oszicar_path):
            ionic_steps = self._parse_oszicar(oszicar_path)
            
            if not ionic_steps:
                result.status = CalculationStatus.CONTINUE
                result.details = "OSZICAR exists but no ionic steps parsed yet"
                return result
            
            result.ionic_steps_completed = len(ionic_steps)
            result.electronic_convergence_history = [s.n_electronic_steps for s in ionic_steps]
            
            # Step 3: Apply heuristic checks
            # Check 3a: Electronic divergence (NELM hit)
            divergence_result = self._check_electronic_divergence(ionic_steps)
            if divergence_result:
                divergence_result.total_elapsed_time = total_elapsed
                divergence_result.time_since_last_check = time_since_check
                divergence_result.ionic_steps_completed = len(ionic_steps)
                divergence_result.electronic_convergence_history = result.electronic_convergence_history
                return divergence_result
            
            # Check 3b: Energy oscillation
            oscillation_result = self._check_energy_oscillation(ionic_steps)
            if oscillation_result:
                oscillation_result.total_elapsed_time = total_elapsed
                oscillation_result.time_since_last_check = time_since_check
                oscillation_result.ionic_steps_completed = len(ionic_steps)
                oscillation_result.electronic_convergence_history = result.electronic_convergence_history
                return oscillation_result
            
            # Check 3c: Stagnation
            stagnation_result = self._check_stagnation(ionic_steps, outcar_path)
            if stagnation_result:
                stagnation_result.total_elapsed_time = total_elapsed
                stagnation_result.time_since_last_check = time_since_check
                stagnation_result.ionic_steps_completed = len(ionic_steps)
                stagnation_result.electronic_convergence_history = result.electronic_convergence_history
                return stagnation_result
            
            # Check 3d: Abnormal physics
            physics_result = self._check_abnormal_physics(ionic_steps, outcar_path)
            if physics_result:
                physics_result.total_elapsed_time = total_elapsed
                physics_result.time_since_last_check = time_since_check
                physics_result.ionic_steps_completed = len(ionic_steps)
                physics_result.electronic_convergence_history = result.electronic_convergence_history
                return physics_result
        
        # All checks passed - continue
        result.status = CalculationStatus.CONTINUE
        result.details = f"Calculation proceeding normally ({result.ionic_steps_completed} ionic steps)"
        return result
    
    def _parse_oszicar(self, oszicar_path: str) -> List[IonicStepData]:
        """
        Parse OSZICAR file to extract ionic step data.
        
        OSZICAR format for each ionic step:
            N       E              dE      d eps       ncg     rms    rms(c)
        DAV:   1    -0.123456E+02   ...
        DAV:   2    -0.123457E+02   ...
        ...
           1 F= -.12345678E+02 E0= -.12345678E+02  d E =-.12345678E+02  mag=   0.0000
        
        Args:
            oszicar_path: Path to OSZICAR file
            
        Returns:
            List of IonicStepData objects
        """
        ionic_steps = []
        
        try:
            with open(oszicar_path, 'r') as f:
                lines = f.readlines()
        except (IOError, OSError) as e:
            logging.warning(f"Could not read OSZICAR: {e}")
            return []
        
        if not lines:
            return []
        
        # Pattern for ionic step summary line
        # Format: " N F= X.XXX E0= X.XXX  d E =X.XXX  mag= X.XXX" (mag optional)
        ionic_pattern = re.compile(
            r'^\s*(\d+)\s+F=\s*([-+]?\d+\.?\d*E?[-+]?\d*)\s+'
            r'E0=\s*([-+]?\d+\.?\d*E?[-+]?\d*)\s+'
            r'd\s*E\s*=\s*([-+]?\d+\.?\d*E?[-+]?\d*)(?:\s+mag=\s*([-+]?\d+\.?\d*))?'
        )
        
        # Pattern for electronic step (DAV/RMM/CG)
        elec_pattern = re.compile(r'^\s*(DAV|RMM|CG):\s+(\d+)')
        
        current_elec_steps = 0
        
        for line in lines:
            # Count electronic steps
            elec_match = elec_pattern.match(line)
            if elec_match:
                step_num = int(elec_match.group(2))
                current_elec_steps = max(current_elec_steps, step_num)
                continue
            
            # Parse ionic step summary
            ionic_match = ionic_pattern.match(line)
            if ionic_match:
                step_num = int(ionic_match.group(1))
                E0 = float(ionic_match.group(3))
                dE = float(ionic_match.group(4))
                mag = float(ionic_match.group(5)) if ionic_match.group(5) else None
                
                ionic_steps.append(IonicStepData(
                    step_number=step_num,
                    n_electronic_steps=current_elec_steps,
                    E0=E0,
                    dE=dE,
                    mag=mag
                ))
                
                # Reset electronic step counter for next ionic step
                current_elec_steps = 0
        
        logging.debug(f"Parsed {len(ionic_steps)} ionic steps from OSZICAR")
        return ionic_steps
    
    def _check_electronic_divergence(
        self,
        ionic_steps: List[IonicStepData],
        consecutive_threshold: int = 3
    ) -> Optional[AssessmentResult]:
        """
        Check if electronic steps are hitting NELM limit.
        
        Criterion: Last N consecutive ionic steps all hit NELM limit.
        
        Args:
            ionic_steps: List of ionic step data
            consecutive_threshold: Number of consecutive NELM hits to trigger kill
            
        Returns:
            AssessmentResult with KILL status if divergence detected, None otherwise
        """
        if len(ionic_steps) < consecutive_threshold:
            return None
        
        recent_steps = ionic_steps[-consecutive_threshold:]
        nelm_hits = [s.n_electronic_steps >= self.nelm_threshold for s in recent_steps]
        
        if all(nelm_hits):
            avg_elec_steps = sum(s.n_electronic_steps for s in recent_steps) / len(recent_steps)
            
            return AssessmentResult(
                status=CalculationStatus.KILL,
                failure_reason=FailureReason.ELECTRONIC_DIVERGENCE,
                suggested_fix=SuggestedFix.SWITCH_ALGO_NORMAL,
                details=f"Electronic divergence: Last {consecutive_threshold} ionic steps "
                        f"hit NELM limit (~{avg_elec_steps:.0f} electronic steps each)",
                metrics={
                    "recent_elec_steps": [s.n_electronic_steps for s in recent_steps],
                    "nelm_threshold": self.nelm_threshold,
                    "consecutive_hits": consecutive_threshold
                }
            )
        
        return None
    
    def _check_energy_oscillation(
        self,
        ionic_steps: List[IonicStepData]
    ) -> Optional[AssessmentResult]:
        """
        Check for energy oscillation without downward trend.
        
        Criterion: Over the oscillation window, energy has no consistent downward trend.
        
        Args:
            ionic_steps: List of ionic step data
            
        Returns:
            AssessmentResult with KILL status if oscillation detected, None otherwise
        """
        if len(ionic_steps) < self.oscillation_window:
            return None
        
        recent_steps = ionic_steps[-self.oscillation_window:]
        energies = [s.E0 for s in recent_steps]
        dEs = [s.dE for s in recent_steps]
        
        # Check for oscillation: count sign changes in dE
        sign_changes = sum(
            1 for i in range(1, len(dEs))
            if dEs[i] * dEs[i-1] < 0  # Sign change
        )
        
        # Calculate overall energy trend
        energy_change = energies[-1] - energies[0]
        avg_abs_dE = sum(abs(dE) for dE in dEs) / len(dEs)
        
        # Oscillation detected if:
        # 1. Many sign changes (> half the window)
        # 2. Energy hasn't decreased overall
        # 3. Average |dE| is significant (not just noise)
        if sign_changes >= self.oscillation_window // 2:
            if energy_change >= 0 or abs(energy_change) < avg_abs_dE:
                return AssessmentResult(
                    status=CalculationStatus.KILL,
                    failure_reason=FailureReason.ENERGY_OSCILLATION,
                    suggested_fix=SuggestedFix.REDUCE_POTIM,
                    details=f"Energy oscillation: {sign_changes} sign changes in last "
                            f"{self.oscillation_window} steps, net energy change: {energy_change:.4f} eV",
                    metrics={
                        "sign_changes": sign_changes,
                        "energy_window_start": energies[0],
                        "energy_window_end": energies[-1],
                        "energy_change": energy_change,
                        "avg_abs_dE": avg_abs_dE,
                        "recent_dEs": dEs
                    }
                )
        
        return None
    
    def _check_stagnation(
        self,
        ionic_steps: List[IonicStepData],
        outcar_path: str
    ) -> Optional[AssessmentResult]:
        """
        Check for stagnation: small dE but forces not converging.
        
        Args:
            ionic_steps: List of ionic step data
            outcar_path: Path to OUTCAR for force extraction
            
        Returns:
            AssessmentResult with KILL status if stagnation detected, None otherwise
        """
        if len(ionic_steps) < self.stagnation_step_threshold:
            return None
        
        recent_steps = ionic_steps[-self.stagnation_step_threshold:]
        
        # Check if all recent dE are below threshold
        small_dE_count = sum(
            1 for s in recent_steps
            if abs(s.dE) < self.stagnation_de_threshold
        )
        
        if small_dE_count < self.stagnation_step_threshold * 0.8:
            # Not enough steps with small dE
            return None
        
        # Check forces from OUTCAR
        if os.path.exists(outcar_path):
            max_force = self._get_max_force(outcar_path)
            ediffg = self._get_ediffg(outcar_path)
            
            if max_force is not None and ediffg is not None:
                # EDIFFG negative means force criterion
                if ediffg < 0 and max_force > abs(ediffg):
                    return AssessmentResult(
                        status=CalculationStatus.KILL,
                        failure_reason=FailureReason.FORCE_STAGNATION,
                        suggested_fix=SuggestedFix.INCREASE_POTIM,
                        details=f"Stagnation: {small_dE_count}/{self.stagnation_step_threshold} "
                                f"steps with |dE| < {self.stagnation_de_threshold} eV, "
                                f"but max force {max_force:.4f} eV/Å > EDIFFG {abs(ediffg):.4f}",
                        metrics={
                            "small_dE_count": small_dE_count,
                            "dE_threshold": self.stagnation_de_threshold,
                            "max_force": max_force,
                            "ediffg": ediffg,
                            "recent_dEs": [s.dE for s in recent_steps]
                        }
                    )
        
        return None
    
    def _check_abnormal_physics(
        self,
        ionic_steps: List[IonicStepData],
        outcar_path: str
    ) -> Optional[AssessmentResult]:
        """
        Check for abnormal physics: positive energies or extreme forces.
        
        Args:
            ionic_steps: List of ionic step data
            outcar_path: Path to OUTCAR
            
        Returns:
            AssessmentResult with KILL status if abnormal physics detected, None otherwise
        """
        if not ionic_steps:
            return None
        
        # Check for positive energies.
        # Allow the first few ionic steps to have positive energy — this is common
        # when the initial structure is poor (e.g., unrelaxed surface, molecule in
        # large vacuum box).  Only kill if positive energy persists after 3 ionic
        # steps, indicating a genuine problem (wrong POSCAR, bad MAGMOM, etc.).
        POSITIVE_ENERGY_GRACE_STEPS = 3
        recent_energy = ionic_steps[-1].E0
        if recent_energy > self.energy_threshold:
            if len(ionic_steps) <= POSITIVE_ENERGY_GRACE_STEPS:
                logging.warning(
                    f"Positive energy {recent_energy:.2f} eV at ionic step {len(ionic_steps)} "
                    f"(grace period: {POSITIVE_ENERGY_GRACE_STEPS} steps) — continuing"
                )
            else:
                return AssessmentResult(
                    status=CalculationStatus.KILL,
                    failure_reason=FailureReason.POSITIVE_ENERGY,
                    suggested_fix=SuggestedFix.SCALE_STRUCTURE,
                    details=f"Abnormal physics: Positive energy {recent_energy:.2f} eV "
                            f"persists after {len(ionic_steps)} ionic steps "
                            f"(threshold: {self.energy_threshold} eV)",
                    metrics={
                        "current_energy": recent_energy,
                        "energy_threshold": self.energy_threshold,
                        "ionic_step": len(ionic_steps)
                    }
                )
        
        # Check for extreme forces (atom overlap)
        if os.path.exists(outcar_path):
            max_force = self._get_max_force(outcar_path)
            
            if max_force is not None and max_force > self.force_threshold:
                return AssessmentResult(
                    status=CalculationStatus.KILL,
                    failure_reason=FailureReason.EXTREME_FORCES,
                    suggested_fix=SuggestedFix.SCALE_STRUCTURE,
                    details=f"Abnormal physics: Extreme force {max_force:.1f} eV/Å detected "
                            f"(threshold: {self.force_threshold} eV/Å). Possible atom overlap.",
                    metrics={
                        "max_force": max_force,
                        "force_threshold": self.force_threshold
                    }
                )
        
        return None
    
    def _check_outcar_errors(self, outcar_path: str) -> AssessmentResult:
        """
        Check OUTCAR for explicit VASP error messages.
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            AssessmentResult (with KILL if error found, CONTINUE otherwise)
        """
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
        except (IOError, OSError) as e:
            logging.warning(f"Could not read OUTCAR: {e}")
            return AssessmentResult(status=CalculationStatus.CONTINUE)
        
        # Map of error patterns to (FailureReason, SuggestedFix, Description)
        error_patterns = [
            # WAVECAR plane-wave basis mismatch (ENCUT changed between steps)
            # Must be checked BEFORE SYMPREC_ERROR because "I REFUSE TO CONTINUE" also
            # triggers that pattern but the root cause here is different.
            (r'while reading WAVECAR.*plane wave coefficients changed',
             FailureReason.WAVECAR_BASIS_CHANGE,
             SuggestedFix.FRESH_START,
             "WAVECAR incompatible: plane wave basis changed between steps"),

            # ZBRENT error (often in cell optimization)
            (r'ZBRENT:\s*fatal\s*error|ZBRENT:\s*\d+\s*step',
             FailureReason.ZBRENT_ERROR,
             SuggestedFix.REDUCE_POTIM,
             "ZBRENT bracketing error in line search"),
            
            # EDDDAV error (diagonalization issues)
            (r'Error\s+EDDDAV:\s*Call\s+to\s*ZHEGV',
             FailureReason.EDDDAV_ERROR,
             SuggestedFix.SWITCH_ALGO_NORMAL,
             "EDDDAV diagonalization error"),
            
            # BRMIX error (charge mixing issues)
            (r'BRMIX:\s*very\s*serious\s*problems',
             FailureReason.BRMIX_ERROR,
             SuggestedFix.REDUCE_AMIX,
             "BRMIX charge density mixing problem"),
            
            # Grid/FFT errors
            (r'internal\s*error\s*in\s*.*FFT|ERROR:\s*SGRCON|VERY\s*BAD\s*NEWS.*internal\s*error',
             FailureReason.GRID_ERROR,
             SuggestedFix.INCREASE_ENCUT,
             "FFT grid or symmetry error"),
            
            # PRICEL error (cell determination)
            (r'PRICEL:\s*internal\s*error|Found\s*some\s*non-integer\s*element',
             FailureReason.PRICEL_ERROR,
             SuggestedFix.SET_ISYM_0,
             "PRICEL primitive cell error"),
            
            # CHGCAR read errors (must be checked BEFORE SYMPREC — both can produce
            # "I REFUSE TO CONTINUE" but the root cause and fix differ)
            (r'charge density could not be read from file CHGCAR|CHGCAR.*incompatible',
             FailureReason.CHGCAR_READ_ERROR,
             SuggestedFix.FRESH_START,
             "CHGCAR incompatible with current k-mesh (ICHARG>=11 requires matching CHGCAR)"),

            # MAGMOM parsing error (bad format, wrong count, or non-numeric text)
            (r'Error reading item MAGMOM',
             FailureReason.MAGMOM_ERROR,
             SuggestedFix.FIX_MAGMOM,
             "MAGMOM format error — value count must match number of atoms in POSCAR"),

            # Hybrid functional + ICHARG>10 conflict
            (r'Fock exchange.*cannot be run.*ICHARG|ICHARG.*Fock exchange',
             FailureReason.HYBRID_ICHARG_CONFLICT,
             SuggestedFix.FIX_HYBRID_RESTART,
             "Hybrid functional (LHFCALC) cannot use ICHARG>10 — must restart from WAVECAR"),

            # ADDGRID + Tkatchenko-Scheffler dispersion conflict
            (r'ADDGRID.*not supported for TS correction',
             FailureReason.ADDGRID_TS_CONFLICT,
             SuggestedFix.DISABLE_ADDGRID,
             "ADDGRID=.TRUE. is incompatible with TS dispersion methods (IVDW=20/21)"),

            # Line-mode KPOINTS used where uniform mesh is required
            (r'uniform k mesh.*cannot be run.*line.mode|requires a uniform k mesh',
             FailureReason.KPOINTS_MODE_CONFLICT,
             SuggestedFix.FIX_KPOINTS_MESH,
             "Calculation requires uniform k-mesh but KPOINTS is in line mode"),

            # NOMEGA exceeds GPU VASP grid point limit
            (r'Currently not more than \d+ grid points supported',
             FailureReason.NOMEGA_LIMIT,
             SuggestedFix.REDUCE_NOMEGA,
             "NOMEGA exceeds maximum supported grid points for this VASP build"),

            # Symmetry/SYMPREC errors
            # IMPORTANT: Do NOT match the bare string "SYMPREC" - it appears in the
            # INCAR echo at the top of every OUTCAR when SYMPREC is explicitly set!
            # Match only actual VASP error messages related to symmetry problems:
            #   - "try changing SYMPREC" (from Bravais lattice mismatch error block)
            #   - "I REFUSE TO CONTINUE" (fatal error termination)
            #   - "Inconsistent Bravais lattice" (the actual error description)
            #   - "POSMAP internal error" (symmetry mapping failure)
            #   - "determinant of rotation" (rotation matrix error)
            (r'try changing SYMPREC|I REFUSE TO CONTINUE|Inconsistent Bravais lattice|POSMAP\s*internal\s*error|determinant\s*of\s*rotation',
             FailureReason.SYMPREC_ERROR,
             SuggestedFix.REDUCE_SYMPREC,
             "Symmetry precision error"),
            
            # Insufficient bands
            (r'VERY\s*BAD\s*NEWS.*your\s*highest\s*band\s*is\s*occupied',
             FailureReason.INSUFFICIENT_BANDS,
             SuggestedFix.INCREASE_NBANDS,
             "Insufficient bands - highest band occupied"),
            
            # Memory errors
            (r'malloc|SIGKILL|out\s*of\s*memory|Cannot\s*allocate',
             FailureReason.MEMORY_ERROR,
             SuggestedFix.INCREASE_MEMORY,
             "Memory allocation error"),
            
            # Sub-space diag
            (r'Sub-Space-Matrix\s*is\s*not\s*hermitian',
             FailureReason.EDDDAV_SUBSPACE_ERROR,
             SuggestedFix.SWITCH_ALGO_ALL,
             "Sub-space matrix error"),
        ]
        
        for pattern, reason, fix, description in error_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logging.warning(f"OUTCAR error detected: {description}")
                return AssessmentResult(
                    status=CalculationStatus.KILL,
                    failure_reason=reason,
                    suggested_fix=fix,
                    details=f"VASP error: {description}"
                )
        
        return AssessmentResult(status=CalculationStatus.CONTINUE)
    
    def _check_convergence_outcar(self, outcar_path: str) -> bool:
        """
        Check if calculation converged successfully from OUTCAR.
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            True if converged, False otherwise
        """
        try:
            with open(outcar_path, 'r') as f:
                # Read last 1000 lines (convergence message at end)
                lines = f.readlines()[-1000:]
            
            content = ''.join(lines)
            
            # Check for convergence indicators
            if "reached required accuracy" in content:
                return True
            if "General timing and accounting informations" in content:
                # Calculation finished - need to check if it was a static calc
                # For static calcs, check from file beginning for IBRION/NSW
                if self._is_static_calculation(outcar_path):
                    return True
            
        except Exception as e:
            logging.debug(f"Error checking convergence: {e}")
        
        return False
    
    def _get_max_force(self, outcar_path: str) -> Optional[float]:
        """
        Extract maximum force from OUTCAR (last ionic step).
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            Maximum force magnitude in eV/Å, or None if not found
        """
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
            
            # Find all TOTAL-FORCE blocks
            force_blocks = re.findall(
                r'TOTAL-FORCE.*?\n-+\n(.*?)\n\s*-+',
                content,
                re.DOTALL
            )
            
            if not force_blocks:
                return None
            
            # Parse the last force block
            last_block = force_blocks[-1]
            max_force = 0.0
            
            for line in last_block.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        fx = float(parts[3])
                        fy = float(parts[4])
                        fz = float(parts[5])
                        force_mag = (fx**2 + fy**2 + fz**2)**0.5
                        max_force = max(max_force, force_mag)
                    except (ValueError, IndexError):
                        continue
            
            return max_force if max_force > 0 else None
            
        except Exception as e:
            logging.debug(f"Error extracting max force: {e}")
            return None
    
    def _get_ediffg(self, outcar_path: str) -> Optional[float]:
        """
        Extract EDIFFG parameter from OUTCAR.
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            EDIFFG value, or None if not found
        """
        try:
            with open(outcar_path, 'r', errors='ignore') as f:
                for line in f:
                    if 'EDIFFG' in line and '=' in line:
                        match = re.search(r'EDIFFG\s*=\s*([-+]?\d+\.?\d*E?[-+]?\d*)', line)
                        if match:
                            return float(match.group(1))
        except Exception as e:
            logging.debug(f"Error extracting EDIFFG: {e}")
        
        return None
    
    def _is_dfpt_calculation(self, outcar_path: str) -> bool:
        """
        Check if this is a DFPT calculation (IBRION=7 or 8).
        
        DFPT calculations (used for phonons, Raman, etc.) don't have ionic steps
        in the traditional sense, so the standard OSZICAR parsing won't work.
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            True if IBRION is 7 or 8, False otherwise
        """
        try:
            with open(outcar_path, 'r', errors='ignore') as f:
                for line in f:
                    if 'IBRION' in line and '=' in line:
                        match = re.search(r'IBRION\s*=\s*(-?\d+)', line)
                        if match:
                            ibrion = int(match.group(1))
                            if ibrion in [7, 8]:
                                logging.debug(f"Detected DFPT calculation (IBRION={ibrion})")
                                return True
                            return False
        except Exception as e:
            logging.debug(f"Error checking IBRION: {e}")
        
        return False
    
    def _is_static_calculation(self, outcar_path: str) -> bool:
        """
        Check if this is a static calculation (NSW=0 or IBRION=-1).
        
        Static calculations (SCF, band structure, DOS) only have electronic steps,
        no ionic steps. The OSZICAR format is different - it only shows electronic
        convergence without ionic step summary lines.
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            True if NSW=0 or IBRION=-1, False otherwise
        """
        try:
            with open(outcar_path, 'r', errors='ignore') as f:
                nsw = None
                ibrion = None
                
                for line in f:
                    # Check NSW
                    if 'NSW' in line and '=' in line and 'PSTRESS' not in line:
                        match = re.search(r'NSW\s*=\s*(\d+)', line)
                        if match:
                            nsw = int(match.group(1))
                    
                    # Check IBRION
                    if 'IBRION' in line and '=' in line:
                        match = re.search(r'IBRION\s*=\s*(-?\d+)', line)
                        if match:
                            ibrion = int(match.group(1))
                    
                    # Early exit once we have both values
                    if nsw is not None and ibrion is not None:
                        break
                
                # Static calculation: NSW=0 or IBRION=-1
                if nsw == 0 or ibrion == -1:
                    logging.debug(f"Detected static calculation (NSW={nsw}, IBRION={ibrion})")
                    return True
                    
        except Exception as e:
            logging.debug(f"Error checking static calculation: {e}")
        
        return False
    
    def _check_static_progress(self, outcar_path: str) -> Dict[str, Any]:
        """
        Check progress of static calculation from OUTCAR/OSZICAR.
        
        Static calculations only have electronic SCF iterations. We track:
        - Current electronic step number
        - Energy convergence (dE)
        - NELM limit for reference
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            Dictionary with progress info (electronic_step, last_energy, last_de, nelm)
        """
        progress = {
            "electronic_step": 0,
            "last_energy": None,
            "last_de": None,
            "nelm": 60,  # Default NELM
        }
        
        calc_dir = os.path.dirname(outcar_path)
        oszicar_path = os.path.join(calc_dir, "OSZICAR")
        
        # Get NELM from OUTCAR
        try:
            with open(outcar_path, 'r', errors='ignore') as f:
                for line in f:
                    if 'NELM' in line and '=' in line and 'NELMIN' not in line and 'NELMDL' not in line:
                        match = re.search(r'NELM\s*=\s*(\d+)', line)
                        if match:
                            progress["nelm"] = int(match.group(1))
                            break
        except Exception as e:
            logging.debug(f"Error reading NELM: {e}")
        
        # Parse OSZICAR for electronic step progress
        try:
            if os.path.exists(oszicar_path):
                with open(oszicar_path, 'r') as f:
                    lines = f.readlines()
                
                # Pattern for electronic step (DAV/RMM/CG)
                # Format: "DAV:   1     0.123456E+02    0.12346E+02   -0.12768E+04 10888   0.532E+02"
                elec_pattern = re.compile(
                    r'^\s*(DAV|RMM|CG):\s+(\d+)\s+([-+]?\d+\.?\d*E?[-+]?\d*)\s+([-+]?\d+\.?\d*E?[-+]?\d*)'
                )
                
                for line in reversed(lines):
                    match = elec_pattern.match(line)
                    if match:
                        progress["electronic_step"] = int(match.group(2))
                        progress["last_energy"] = float(match.group(3))
                        progress["last_de"] = float(match.group(4))
                        break
                        
        except Exception as e:
            logging.debug(f"Error parsing OSZICAR for static progress: {e}")
        
        return progress
    
    def _check_dfpt_progress(self, outcar_path: str) -> Dict[str, Any]:
        """
        Check progress of DFPT calculation from OUTCAR.
        
        DFPT calculations (IBRION=7,8) report progress differently - they work through
        degrees of freedom (perturbations) rather than ionic steps.
        
        VASP output patterns:
        - "Degree of freedom:   N/ M" - main progress indicator for phonon/Raman
        - "Linear response progress:" - section header
        - "Direction:   N" - direction within linear response
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            Dictionary with progress info (perturbations_done, total_perturbations, etc.)
        """
        progress = {
            "perturbations_done": 0,
            "total_perturbations": 0,
            "electronic_steps_current": 0,
            "last_energy": None,
            "directions_done": 0,
        }
        
        try:
            with open(outcar_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Primary pattern: "Degree of freedom:   N/ M" (IBRION=8 phonon/Raman)
            # This is the main progress indicator for DFPT calculations
            dof_matches = re.findall(
                r'Degree\s+of\s+freedom:\s+(\d+)/\s*(\d+)',
                content
            )
            if dof_matches:
                # Get the latest progress (last match gives current state)
                last_done, total = dof_matches[-1]
                progress["perturbations_done"] = int(last_done)
                progress["total_perturbations"] = int(total)
            
            # Secondary pattern: Count "Direction:" markers for linear response phases
            # This appears in the electric field response calculations
            direction_matches = re.findall(r'Direction:\s+(\d+)', content)
            if direction_matches:
                progress["directions_done"] = len(direction_matches)
            
            # Fallback: Count "Linear response progress:" sections
            if progress["perturbations_done"] == 0:
                lr_progress_count = len(re.findall(r'Linear\s+response\s+progress:', content))
                if lr_progress_count > 0:
                    progress["perturbations_done"] = lr_progress_count
            
            # Get current electronic step count from last DAV/RMM block
            elec_matches = re.findall(r'(DAV|RMM|CG):\s+(\d+)', content)
            if elec_matches:
                progress["electronic_steps_current"] = int(elec_matches[-1][1])
            
            # Get last energy
            energy_matches = re.findall(
                r'free\s+energy\s+TOTEN\s*=\s*([-+]?\d+\.?\d*)',
                content
            )
            if energy_matches:
                progress["last_energy"] = float(energy_matches[-1])
                
        except Exception as e:
            logging.debug(f"Error checking DFPT progress: {e}")
        
        return progress
    
    def _is_soc_calculation(self, outcar_path: str) -> bool:
        """
        Check if this is a SOC calculation (LSORBIT=.TRUE.).
        
        SOC calculations have special characteristics:
        - LSORBIT = .TRUE.
        - Usually LNONCOLLINEAR = .TRUE.
        - May have SAXIS specification
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            True if LSORBIT is .TRUE., False otherwise
        """
        try:
            with open(outcar_path, 'r', errors='ignore') as f:
                for line in f:
                    if 'LSORBIT' in line and '=' in line:
                        # Check for "LSORBIT = T" or "LSORBIT =     T"
                        if re.search(r'LSORBIT\s*=\s*T', line, re.IGNORECASE):
                            logging.debug("Detected SOC calculation (LSORBIT=.TRUE.)")
                            return True
                        return False
        except Exception as e:
            logging.debug(f"Error checking LSORBIT: {e}")
        
        return False
    
    def _check_soc_progress(self, outcar_path: str, calc_dir: str) -> Dict[str, Any]:
        """
        Check progress of SOC calculation from OUTCAR/OSZICAR/print-out.
        
        SOC calculations are typically static (NSW=0) but may take much longer
        due to spinor basis (2x bands) and non-collinear magnetism.
        
        We track:
        - Current electronic step number
        - Energy convergence (dE)
        - SAXIS direction
        - Whether SCF has converged (important milestone)
        
        Note: For static SOC calculations, OSZICAR may be empty until convergence.
        We also check print-out file for real-time electronic step progress.
        
        Args:
            outcar_path: Path to OUTCAR file
            calc_dir: Directory containing the calculation
            
        Returns:
            Dictionary with progress info (electronic_step, last_energy, last_de, 
                                          nelm, saxis, scf_converged)
        """
        progress = {
            "electronic_step": 0,
            "last_energy": None,
            "last_de": None,
            "nelm": 60,  # Default NELM
            "saxis": "unknown",
            "scf_converged": False,
            "lnoncollinear": False,
        }
        
        oszicar_path = os.path.join(calc_dir, "OSZICAR")
        printout_path = os.path.join(calc_dir, "print-out")
        
        # Get NELM and SAXIS from OUTCAR
        try:
            with open(outcar_path, 'r', errors='ignore') as f:
                for line in f:
                    # Check NELM
                    if 'NELM' in line and '=' in line and 'NELMIN' not in line and 'NELMDL' not in line:
                        match = re.search(r'NELM\s*=\s*(\d+)', line)
                        if match:
                            progress["nelm"] = int(match.group(1))
                    
                    # Check SAXIS
                    if 'SAXIS' in line and '=' in line:
                        match = re.search(r'SAXIS\s*=\s*([\d\s\.\-]+)', line)
                        if match:
                            progress["saxis"] = match.group(1).strip()
                    
                    # Check LNONCOLLINEAR
                    if 'LNONCOLLINEAR' in line and '=' in line:
                        if re.search(r'LNONCOLLINEAR\s*=\s*T', line, re.IGNORECASE):
                            progress["lnoncollinear"] = True
        except Exception as e:
            logging.debug(f"Error reading SOC parameters: {e}")
        
        # Pattern for electronic step (DAV/RMM/CG)
        # Format: "DAV:   1     0.123456E+02    0.12346E+02   -0.12768E+04 10888   0.532E+02"
        elec_pattern = re.compile(
            r'^\s*(DAV|RMM|CG):\s+(\d+)\s+([-+]?\d+\.?\d*E?[-+]?\d*)\s+([-+]?\d+\.?\d*E?[-+]?\d*)'
        )
        
        # Try OSZICAR first
        try:
            if os.path.exists(oszicar_path) and os.path.getsize(oszicar_path) > 0:
                with open(oszicar_path, 'r') as f:
                    lines = f.readlines()
                
                for line in reversed(lines):
                    match = elec_pattern.match(line)
                    if match:
                        progress["electronic_step"] = int(match.group(2))
                        progress["last_energy"] = float(match.group(3))
                        progress["last_de"] = float(match.group(4))
                        break
        except Exception as e:
            logging.debug(f"Error parsing OSZICAR for SOC progress: {e}")
        
        # If OSZICAR is empty, try print-out file (real-time VASP output)
        # This is important for static SOC calculations where OSZICAR may not be written
        if progress["electronic_step"] == 0:
            try:
                if os.path.exists(printout_path):
                    with open(printout_path, 'r', errors='ignore') as f:
                        # Read last 50KB of print-out for efficiency
                        f.seek(0, 2)
                        file_size = f.tell()
                        read_size = min(50000, file_size)
                        f.seek(max(0, file_size - read_size))
                        content = f.read()
                    
                    # Find all electronic steps in the content
                    for line in content.split('\n'):
                        match = elec_pattern.match(line)
                        if match:
                            step = int(match.group(2))
                            if step > progress["electronic_step"]:
                                progress["electronic_step"] = step
                                progress["last_energy"] = float(match.group(3))
                                progress["last_de"] = float(match.group(4))
                    
                    if progress["electronic_step"] > 0:
                        logging.debug(f"SOC progress from print-out: step {progress['electronic_step']}")
            except Exception as e:
                logging.debug(f"Error parsing print-out for SOC progress: {e}")
        
        # Check if SCF converged by looking at OUTCAR
        try:
            with open(outcar_path, 'r', errors='ignore') as f:
                # Read last part of file for convergence check
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                # Read last 100KB
                read_size = min(100000, file_size)
                f.seek(max(0, file_size - read_size))
                content = f.read()
                
                # Check for convergence indicators
                if "reached required accuracy" in content:
                    progress["scf_converged"] = True
                elif "General timing and accounting" in content:
                    # Calculation finished
                    progress["scf_converged"] = True
                    
        except Exception as e:
            logging.debug(f"Error checking SOC convergence: {e}")
        
        return progress


def create_rule_monitor(**kwargs) -> RuleMonitor:
    """
    Factory function to create a RuleMonitor with optional custom parameters.
    
    Args:
        **kwargs: Parameters to override defaults
        
    Returns:
        Configured RuleMonitor instance
    """
    return RuleMonitor(**kwargs)
