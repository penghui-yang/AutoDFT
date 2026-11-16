"""
Execution-focused VASP executor.
"""
from __future__ import annotations

import os
import shutil
import logging
import re
from typing import Optional, Dict, Any, List
from pathlib import Path

from common_types.workflow_models import (
    ExecutionContext,
    StepExecutionResult,
    StepOutcomeSummary,
    WorkflowStep,
)
from execution.job_manager import run_command, run_vasp_job, JobTimeoutError, JobTerminatedByRuleMonitorError
from config import MONITOR_CHECK_INTERVAL
from tools.file_parsers import (
    detect_symmetry_change,
    estimate_bandgap_from_eigenval,
    extract_cell_volume,
    extract_electronic_convergence,
    extract_max_force,
    extract_oszicar_tail,
    extract_outcar_summary,
    extract_space_group,
    extract_stress_tensor,
    extract_total_energy,
    get_formula_from_poscar,
)
from tools.postprocessing.neb import interpolate_images


class VASPExecutor:
    """
    Flexible executor agent that runs workflow steps based on detailed specifications.
    
    This agent can execute any VASP calculation defined in a WorkflowStep,
    without hardcoded step types.
    """
    
    def __init__(self, work_base_dir: str, config: Optional[Dict[str, Any]] = None, llm_client: Any = None):
        """
        Initialize the executor.

        Args:
            work_base_dir: Base directory for all calculations
            config: Run configuration (forwarded to the runtime monitor)
            llm_client: Shared LLM client, so monitor calls count toward the workflow token budget
        """
        self.work_base_dir = work_base_dir
        self.config = config or {}
        self.llm_client = llm_client
        logging.info(f"VASPExecutor initialized with work_base_dir={work_base_dir}")

    def execute_step(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        step_index: int,
        timeout_hours: float = None,
        enable_monitoring: bool = True
    ) -> StepExecutionResult:
        """
        Execute a single workflow step.
        
        This method:
        1. Creates working directory
        2. Prepares input files (POSCAR, POTCAR, INCAR, KPOINTS)
        3. Executes VASPKIT actions
        4. Submits VASP job
        5. Validates outputs
        6. Performs sanity checks
        
        Args:
            step: WorkflowStep to execute
            context: Current execution context
            step_index: Index of this step in the workflow
            timeout_hours: Maximum RUNNING time in hours for VASP job 
                          (excludes pending/queue time). None means no timeout.
            enable_monitoring: Enable rule-based/LLM monitoring for VASP job execution
            
        Returns:
            StepExecutionResult with execution status and outputs
        """
        logging.info(f"[{context.material_id}] Executing step {step_index}: {step.name}")
        logging.info(f"  Description: {step.description}")
        if timeout_hours:
            logging.info(f"  Timeout: {timeout_hours} hours (running time only, excluding queue wait)")
        
        # Create work directory
        work_dir = os.path.join(
            self.work_base_dir,
            context.material_id,
            f"{step_index:02d}_{step.name.replace('/', '-')}"
        )
        os.makedirs(work_dir, exist_ok=True)
        logging.info(f"  Working directory: {work_dir}")
        
        # Track timing info
        timing_info = None
        
        try:
            # Step 1: Prepare input files
            self._prepare_input_files(step, context, work_dir, step_index)
            
            # Step 2: Execute VASPKIT actions
            self._execute_vaspkit_actions(step, work_dir)
            
            # Step 3: Submit VASP job (only if this is a computing step)
            if step.agent_type == "computing" and step.INCAR:
                # Remove stale VASP output files from any previous attempt in the
                # same working directory.  Without this, the RuleMonitor reads the
                # old OUTCAR at job start and immediately kills the new job before
                # VASP has a chance to overwrite it.
                for stale in ("OUTCAR", "OSZICAR", "vasprun.xml"):
                    stale_path = os.path.join(work_dir, stale)
                    if os.path.exists(stale_path):
                        os.remove(stale_path)
                        logging.debug(f"    Removed stale {stale} before retry")

                job_name = f"{context.material_id}_{step.name}"
                timing_info = run_vasp_job(
                    job_name, work_dir,
                    timeout_hours=timeout_hours,
                    enable_monitoring=enable_monitoring,
                    monitor_interval=self.config.get("MONITOR_CHECK_INTERVAL", MONITOR_CHECK_INTERVAL),
                    monitor_config=self.config,
                    llm_client=self.llm_client,
                    physical_context={
                        "material_id": context.material_id,
                        "formula": context.formula,
                        "material_type": (context.metadata or {}).get("material_type"),
                        "step_name": step.name,
                        "step_objective": step.description,
                    },
                )
                # Treat non-zero exit code as a hard failure so the recovery
                # path (MAX_RETRIES_PER_STEP) can handle it instead of the
                # reflector looping forever on a step with no output.
                exit_code = timing_info.get('exit_code', 0) if timing_info else 0
                if exit_code != 0:
                    # Extract the real VASP error from OUTCAR before reporting
                    from agents.rule_monitor import RuleMonitor
                    vasp_error_detail = self._extract_vasp_error(work_dir)
                    outcar_path = os.path.join(work_dir, "OUTCAR")
                    failure_reason = None
                    suggested_fix = None
                    if os.path.exists(outcar_path):
                        try:
                            monitor = RuleMonitor()
                            assessment = monitor._check_outcar_errors(outcar_path)
                            if assessment.failure_reason and assessment.failure_reason.value != "NONE":
                                failure_reason = assessment.failure_reason.value
                                suggested_fix = assessment.suggested_fix.value if assessment.suggested_fix else None
                                vasp_error_detail = vasp_error_detail or assessment.details or ""
                        except Exception:
                            pass

                    error_msg = (
                        f"VASP process exited with non-zero code {exit_code}"
                        + (f" (signal {exit_code - 128})" if exit_code > 128 else "")
                    )
                    if vasp_error_detail:
                        error_msg += f". VASP error: {vasp_error_detail}"
                    else:
                        error_msg += ". The binary may have crashed or be incompatible with this CPU."

                    result = StepExecutionResult(
                        step_name=step.name,
                        status="failed",
                        output_dir=work_dir,
                        error_message=error_msg,
                    )
                    if timing_info:
                        result.metrics['pending_time'] = timing_info.get('pending_time', 0)
                        result.metrics['running_time'] = timing_info.get('running_time', 0)
                        result.metrics['total_time'] = timing_info.get('total_time', 0)
                    result.metrics['exit_code'] = exit_code
                    if failure_reason:
                        result.metrics['failure_reason'] = failure_reason
                    if suggested_fix:
                        result.metrics['suggested_fix'] = suggested_fix
                    if vasp_error_detail:
                        result.metrics['vasp_error_detail'] = vasp_error_detail
                    return result

            # Step 4: Validate outputs
            outputs_produced = self._validate_outputs(step, work_dir)
            
            # Step 5: Perform sanity checks
            warnings = self._perform_sanity_checks(step, work_dir)
            
            # Step 6: Extract data if postprocessing step
            extracted_data = {}
            if step.agent_type == "postprocessing":
                extracted_data = self._extract_data(step, work_dir)
            
            logging.info(f"[{context.material_id}] Step {step.name} completed successfully")
            
            result = StepExecutionResult(
                step_name=step.name,
                status="success",
                output_dir=work_dir,
                outputs_produced=outputs_produced,
                extracted_data=extracted_data,
                warnings=warnings
            )
            
            # Store timing info in metrics
            if timing_info:
                result.metrics['pending_time'] = timing_info['pending_time']
                result.metrics['running_time'] = timing_info['running_time']
                result.metrics['total_time'] = timing_info['total_time']
            
            return result
            
        except JobTimeoutError as e:
            logging.error(f"[{context.material_id}] Step {step.name} timed out: {e}")
            result = StepExecutionResult(
                step_name=step.name,
                status="timeout",
                output_dir=work_dir,
                error_message=str(e)
            )
            # Store timing info from the timeout exception
            if hasattr(e, 'timing_info') and e.timing_info:
                result.metrics['pending_time'] = e.timing_info['pending_time']
                result.metrics['running_time'] = e.timing_info['running_time']
                result.metrics['total_time'] = e.timing_info['total_time']
                result.metrics['timed_out'] = True
            return result
        
        except JobTerminatedByRuleMonitorError as e:
            logging.error(f"[{context.material_id}] Step {step.name} terminated by rule monitor: {e}")
            result = StepExecutionResult(
                step_name=step.name,
                status="terminated_by_rule_monitor",
                output_dir=work_dir,
                error_message=str(e)
            )
            # Store timing info and assessment from the monitor exception
            if hasattr(e, 'timing_info') and e.timing_info:
                result.metrics['pending_time'] = e.timing_info['pending_time']
                result.metrics['running_time'] = e.timing_info['running_time']
                result.metrics['total_time'] = e.timing_info['total_time']
                result.metrics['terminated_by_rule_monitor'] = True
            if hasattr(e, 'assessment') and e.assessment:
                result.metrics['monitor_assessment'] = e.assessment
                result.metrics['failure_reason'] = e.assessment.get('failure_reason')
                result.metrics['suggested_fix'] = e.assessment.get('suggested_fix')
            return result
            
        except Exception as e:
            logging.error(f"[{context.material_id}] Step {step.name} failed: {e}")
            return StepExecutionResult(
                step_name=step.name,
                status="failed",
                output_dir=work_dir,
                error_message=str(e)
            )
    
    def _prepare_input_files(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        work_dir: str,
        step_index: int
    ):
        """
        Prepare all required input files for this step.
        
        Args:
            step: WorkflowStep definition
            context: Execution context
            work_dir: Working directory
            step_index: Index of this step in the workflow (0-based)
        """
        logging.info(f"  Preparing input files for {step.name}")
        
        # Check if this is a NEB calculation
        is_neb = self._is_neb_calculation(step)
        
        if is_neb:
            # NEB needs special directory structure
            self._prepare_neb_directories(step, context, work_dir)
        else:
            # Standard POSCAR preparation
            self._prepare_poscar(step, context, work_dir)
        
        # Prepare POTCAR
        self._prepare_potcar(step, context, work_dir)
        
        # Prepare INCAR
        if step.INCAR:
            self._prepare_incar(step, context, work_dir)
        
        # Prepare KPOINTS
        if step.KPOINTS:
            self._prepare_kpoints(step, context, work_dir)
        
        # Fix 3A: Auto-add CHGCAR to input_files_needed when ICHARG >= 10
        if step.INCAR:
            try:
                icharg_val = int(str(step.INCAR.get("ICHARG", 2)))
            except (ValueError, TypeError):
                icharg_val = 2
            if icharg_val >= 10 and "CHGCAR" not in step.input_files_needed:
                step.input_files_needed.append("CHGCAR")
                logging.info(f"    Auto-adding CHGCAR to input_files_needed (ICHARG={icharg_val})")

        # Copy additional files if needed
        self._copy_additional_files(step, context, work_dir)

        # Fix 3B: Pre-flight validation — fail fast if CHGCAR is required but missing/empty
        if "CHGCAR" in step.input_files_needed:
            chgcar_path = os.path.join(work_dir, "CHGCAR")
            if not os.path.exists(chgcar_path) or os.path.getsize(chgcar_path) == 0:
                raise FileNotFoundError(
                    f"Step '{step.name}' requires CHGCAR (ICHARG >= 10) but it is missing or empty "
                    f"in {work_dir}. Ensure the preceding SCF/relaxation step has LCHARG = .TRUE."
                )

            # Fix 3C: CHGCAR k-mesh compatibility check for ICHARG >= 11.
            # ICHARG=11 reads CHGCAR without updating it, so the CHGCAR must come
            # from a step with a compatible k-mesh (same grid dimensions).
            # If mismatched, search earlier steps for a compatible CHGCAR.
            if icharg_val >= 11:
                self._validate_chgcar_compatibility(step, context, work_dir)
    
    def _read_kpoints_grid(self, kpoints_path: str) -> Optional[str]:
        """Read the k-mesh grid line from a KPOINTS file (line 4 for automatic mesh)."""
        try:
            with open(kpoints_path, 'r') as f:
                lines = f.readlines()
            if len(lines) >= 4:
                # Line 3 (0-indexed) is the grid for Monkhorst-Pack / Gamma
                return lines[3].strip()
        except Exception:
            pass
        return None

    def _validate_chgcar_compatibility(
        self, step: WorkflowStep, context: ExecutionContext, work_dir: str
    ):
        """
        Validate that the copied CHGCAR is compatible with the current KPOINTS.

        When ICHARG >= 11, VASP reads charge density from CHGCAR without updating it.
        If the CHGCAR was generated with a different k-mesh, VASP will fail with
        "charge density could not be read from file CHGCAR for ICHARG>10".

        If incompatible, search earlier SCF steps for a CHGCAR with matching k-grid.
        """
        current_kpoints = os.path.join(work_dir, "KPOINTS")
        current_grid = self._read_kpoints_grid(current_kpoints)
        if not current_grid:
            return  # Can't validate without KPOINTS

        chgcar_path = os.path.join(work_dir, "CHGCAR")
        chgcar_size = os.path.getsize(chgcar_path) if os.path.exists(chgcar_path) else 0

        # Search completed VASP steps for CHGCAR with compatible k-mesh
        # (reverse order = most recent first)
        for step_name, step_dir in reversed(list(context.vasp_completed_steps.items())):
            prev_kpoints = os.path.join(step_dir, "KPOINTS")
            prev_grid = self._read_kpoints_grid(prev_kpoints)
            if prev_grid == current_grid:
                prev_chgcar = os.path.join(step_dir, "CHGCAR")
                if os.path.exists(prev_chgcar) and os.path.getsize(prev_chgcar) > 0:
                    prev_size = os.path.getsize(prev_chgcar)
                    if prev_size != chgcar_size:
                        # Found a compatible CHGCAR with different size → replace
                        logging.warning(
                            f"    CHGCAR k-mesh mismatch detected! Current CHGCAR size={chgcar_size}, "
                            f"compatible CHGCAR from '{step_name}' size={prev_size}. Replacing."
                        )
                        shutil.copy(prev_chgcar, chgcar_path)
                    return  # Compatible source found (same grid), done

        # If no compatible source found, log a warning (pre-flight in Fix 3B will catch if empty)
        logging.warning(
            f"    No CHGCAR with matching k-grid ({current_grid}) found in previous steps. "
            f"VASP may fail if ICHARG >= 11."
        )

    def _is_neb_calculation(self, step: WorkflowStep) -> bool:
        """Check if this step is a NEB calculation."""
        # Check step name
        step_name_lower = step.name.lower()
        if "neb" in step_name_lower:
            return True
        
        # Check INCAR for IMAGES parameter
        if step.INCAR:
            images = step.INCAR.get("IMAGES", 0)
            if images and int(images) > 0:
                return True
        
        return False
    
    def _prepare_neb_directories(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        work_dir: str
    ):
        """
        Prepare NEB directory structure with image POSCARs.
        
        NEB requires:
        - 00/POSCAR (initial state)
        - 01/POSCAR ... 0N/POSCAR (intermediate images)
        - 0(N+1)/POSCAR (final state)
        
        Args:
            step: WorkflowStep definition
            context: Execution context
            work_dir: Working directory
        """
        logging.info(f"    Setting up NEB directory structure")
        
        # Get number of images from INCAR
        n_images = int(step.INCAR.get("IMAGES", 5))
        logging.info(f"    Number of intermediate images: {n_images}")
        
        # Find initial and final state POSCARs
        initial_poscar = None
        final_poscar = None
        
        # Look for relaxed structures in completed steps
        for step_name, step_dir in context.vasp_completed_steps.items():
            step_name_lower = step_name.lower()

            # Check for initial state
            if "initial" in step_name_lower or "relax_initial" in step_name_lower:
                contcar = os.path.join(step_dir, "CONTCAR")
                if os.path.exists(contcar) and os.path.getsize(contcar) > 0:
                    initial_poscar = contcar
                    logging.info(f"    Found initial state: {step_name}")

            # Check for final state
            if "final" in step_name_lower or "relax_final" in step_name_lower:
                contcar = os.path.join(step_dir, "CONTCAR")
                if os.path.exists(contcar) and os.path.getsize(contcar) > 0:
                    final_poscar = contcar
                    logging.info(f"    Found final state: {step_name}")

        # If we don't have both states, try to generate them
        if not initial_poscar or not final_poscar:
            # Fall back to using the same structure (won't work for real NEB)
            logging.warning("    Could not find both initial and final states for NEB")
            logging.warning("    NEB requires relaxed initial and final state structures")

            # Try to find any valid structure to at least create the directory structure
            for step_name, step_dir in reversed(list(context.vasp_completed_steps.items())):
                contcar = os.path.join(step_dir, "CONTCAR")
                if os.path.exists(contcar) and os.path.getsize(contcar) > 0:
                    if not initial_poscar:
                        initial_poscar = contcar
                    if not final_poscar:
                        final_poscar = contcar
                    break
        
        if not initial_poscar or not final_poscar:
            raise FileNotFoundError(
                "NEB requires both initial and final state structures. "
                "Make sure to run relaxation steps for both states before NEB."
            )
        
        # Create NEB image directories and interpolate
        try:
            image_files = interpolate_images(
                initial_poscar=initial_poscar,
                final_poscar=final_poscar,
                n_images=n_images,
                output_dir=work_dir,
                method="linear"
            )
            
            if image_files:
                logging.info(f"    Created {len(image_files)} NEB image directories")
            else:
                raise RuntimeError("Failed to generate NEB images")
                
        except Exception as e:
            logging.error(f"    NEB image generation failed: {e}")
            # Create directories manually as fallback
            self._create_neb_directories_fallback(
                work_dir, n_images, initial_poscar, final_poscar
            )
    
    def _create_neb_directories_fallback(
        self,
        work_dir: str,
        n_images: int,
        initial_poscar: str,
        final_poscar: str
    ):
        """
        Create NEB directories with simple linear interpolation as fallback.
        
        Args:
            work_dir: NEB working directory
            n_images: Number of intermediate images
            initial_poscar: Path to initial state POSCAR
            final_poscar: Path to final state POSCAR
        """
        import numpy as np
        
        logging.info(f"    Creating NEB directories with manual interpolation")
        
        # Read initial and final structures
        with open(initial_poscar, 'r') as f:
            initial_lines = f.readlines()
        with open(final_poscar, 'r') as f:
            final_lines = f.readlines()
        
        # Parse positions (simplified - assumes direct coordinates)
        def parse_positions(lines):
            # Find number of atoms
            try:
                atom_counts = [int(x) for x in lines[6].split()]
            except:
                atom_counts = [int(x) for x in lines[5].split()]
            n_atoms = sum(atom_counts)
            
            # Find start of coordinates
            start_line = 7
            if lines[7].strip()[0].upper() in ['S', 'D', 'C']:
                start_line = 8
            if lines[start_line].strip()[0].upper() in ['D', 'C']:
                start_line += 1
            
            positions = []
            for i in range(n_atoms):
                parts = lines[start_line + i].split()
                pos = [float(parts[j]) for j in range(3)]
                positions.append(pos)
            
            return np.array(positions), start_line
        
        try:
            initial_pos, start_line = parse_positions(initial_lines)
            final_pos, _ = parse_positions(final_lines)
        except Exception as e:
            logging.error(f"    Failed to parse POSCAR files: {e}")
            # Just create empty directories
            for i in range(n_images + 2):
                image_dir = os.path.join(work_dir, f"{i:02d}")
                os.makedirs(image_dir, exist_ok=True)
                if i == 0:
                    shutil.copy(initial_poscar, os.path.join(image_dir, "POSCAR"))
                elif i == n_images + 1:
                    shutil.copy(final_poscar, os.path.join(image_dir, "POSCAR"))
            return
        
        # Create directories and interpolate
        for i in range(n_images + 2):
            image_dir = os.path.join(work_dir, f"{i:02d}")
            os.makedirs(image_dir, exist_ok=True)
            
            # Calculate interpolated positions
            fraction = i / (n_images + 1)
            interp_pos = initial_pos + fraction * (final_pos - initial_pos)
            
            # Write POSCAR
            poscar_path = os.path.join(image_dir, "POSCAR")
            with open(poscar_path, 'w') as f:
                # Write header
                for j in range(start_line):
                    f.write(initial_lines[j])
                
                # Write interpolated positions
                for pos in interp_pos:
                    f.write(f"  {pos[0]:.16f}  {pos[1]:.16f}  {pos[2]:.16f}\n")
        
        logging.info(f"    Created {n_images + 2} NEB image directories")
    
    def _prepare_poscar(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        work_dir: str
    ):
        """Prepare POSCAR file.
        
        Tries to find a valid structure file in this order:
        1. CONTCAR from dependent steps (if non-empty)
        2. POSCAR from dependent steps (fallback if CONTCAR is empty, e.g., NSW=0 static calc)
        3. CONTCAR/POSCAR from previous step (backward compatibility)
        4. Generate from CIF (first step)
        """
        
        def _get_valid_structure_file(output_dir: str) -> tuple[str, str] | None:
            """Get a valid (non-empty) structure file from a calculation directory.
            
            Returns (file_path, file_type) or None if no valid file found.
            Prefers CONTCAR, falls back to POSCAR if CONTCAR is empty.
            """
            contcar_path = os.path.join(output_dir, "CONTCAR")
            poscar_path = os.path.join(output_dir, "POSCAR")
            
            # Try CONTCAR first (preferred - contains relaxed structure)
            if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
                return (contcar_path, "CONTCAR")
            
            # Fall back to POSCAR if CONTCAR is empty or missing
            # This handles NSW=0 static calculations that don't update CONTCAR
            if os.path.exists(poscar_path) and os.path.getsize(poscar_path) > 0:
                return (poscar_path, "POSCAR")
            
            return None
        
        # Retry after a failure: start from the archived run (CONTCAR if valid).
        if step.additional_params.get("restart_from_archive") and self._restore_structure_from_archive(work_dir):
            return

        # Check if we need structure from a previous step
        if "CONTCAR" in step.input_files_needed or step.depends_on:
            # Find the most recent completed step that has a valid structure file
            for dep_step_name in reversed(step.depends_on):
                if dep_step_name in context.completed_steps:
                    dep_output_dir = context.completed_steps[dep_step_name]
                    result = _get_valid_structure_file(dep_output_dir)
                    if result:
                        src_path, file_type = result
                        shutil.copy(src_path, os.path.join(work_dir, "POSCAR"))
                        logging.info(f"    Copied {file_type} from {dep_step_name} as POSCAR")
                        return
            
            # If no valid file from dependencies, check completed VASP steps in reverse order
            if context.vasp_completed_steps:
                for step_name in reversed(list(context.vasp_completed_steps.keys())):
                    step_dir = context.vasp_completed_steps[step_name]
                    result = _get_valid_structure_file(step_dir)
                    if result:
                        src_path, file_type = result
                        shutil.copy(src_path, os.path.join(work_dir, "POSCAR"))
                        logging.info(f"    Copied {file_type} from {step_name} as POSCAR (fallback)")
                        return
        
        # Otherwise, generate from CIF (first step)
        if context.cif_path and os.path.exists(context.cif_path):
            cif_file = os.path.basename(context.cif_path)
            shutil.copy(context.cif_path, work_dir)
            # vaspkit requires interactive input even with -file flag
            # Use printf to provide: 1) CIF filename, 2) empty line for default element order
            run_command(f'printf "{cif_file}\\n\\n" | vaspkit -task 105', cwd=work_dir, shell=True)
            logging.info(f"    Generated POSCAR from CIF")
        else:
            raise FileNotFoundError(f"Cannot prepare POSCAR: no CIF or valid previous structure found")
    
    @staticmethod
    def _restore_structure_from_archive(work_dir: str) -> bool:
        """Initialise POSCAR for a retry from the most recent attempt_* archive."""
        from agents.rule_recovery_handler import RuleRecoveryHandler

        handler = RuleRecoveryHandler(work_dir)
        poscar_path = os.path.join(work_dir, "POSCAR")
        if os.path.exists(poscar_path) and handler._is_valid_poscar(poscar_path):
            # Rule-based recovery has already restored (and possibly rescaled) POSCAR.
            logging.info("    Using POSCAR restored by recovery")
            return True
        archives = [
            os.path.join(work_dir, d) for d in os.listdir(work_dir)
            if d.startswith("attempt_") and os.path.isdir(os.path.join(work_dir, d))
        ]
        if not archives:
            return False
        latest = max(archives, key=os.path.getmtime)
        try:
            handler._restore_poscar(latest, use_contcar=handler.config.restart_from_contcar)
            return True
        except FileNotFoundError:
            return False

    def _prepare_potcar(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        work_dir: str
    ):
        """
        Prepare POTCAR file.
        """
        # Check if POTCAR exists in a previous VASP step (skip current work_dir to handle redo case)
        if context.vasp_completed_steps:
            for step_dir in reversed(list(context.vasp_completed_steps.values())):
                if os.path.realpath(step_dir) == os.path.realpath(work_dir):
                    continue
                potcar_path = os.path.join(step_dir, "POTCAR")
                if os.path.exists(potcar_path):
                    shutil.copy(potcar_path, work_dir)
                    logging.info(f"    Copied POTCAR from previous VASP step")
                    return
        
        # Try to generate POTCAR using vaspkit command-line mode
        run_command("vaspkit -task 103", cwd=work_dir, shell=True)
        
        # Verify POTCAR was generated
        potcar_path = os.path.join(work_dir, "POTCAR")
        if os.path.exists(potcar_path) and os.path.getsize(potcar_path) > 0:
            logging.info(f"    Generated POTCAR using vaspkit")
        else:
            raise FileNotFoundError(
                f"Failed to generate POTCAR. Please check vaspkit configuration "
                f"(PBE_PATH in ~/.vaspkit)."
            )
    
    def _prepare_incar(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        work_dir: str
    ):
        """
        Prepare INCAR file from step specifications.
        
        The INCAR dict in step.INCAR contains parameter: value pairs.
        We also merge with global settings.
        """
        incar_params = {}
        
        # Start with global settings
        if context.global_settings:
            gs = context.global_settings
            if gs.ENCUT: incar_params["ENCUT"] = gs.ENCUT
            if gs.PREC: incar_params["PREC"] = gs.PREC
            if gs.EDIFF: incar_params["EDIFF"] = gs.EDIFF
            if gs.EDIFFG: incar_params["EDIFFG"] = gs.EDIFFG
            if gs.ISPIN: incar_params["ISPIN"] = gs.ISPIN
            if gs.ALGO: incar_params["ALGO"] = gs.ALGO
            if gs.LREAL is not None: incar_params["LREAL"] = gs.LREAL
            incar_params.update(gs.additional_settings)
        
        # Override with step-specific parameters
        for key, value in step.INCAR.items():
            # Validate and filter invalid parameters
            if self._is_valid_incar_value(key, value):
                incar_params[key] = value
            else:
                logging.warning(f"    Skipping invalid INCAR parameter: {key} = {value}")
        
        # Fill spin-polarization defaults that the step parameters left unspecified
        self._auto_set_spin_magmom(incar_params, work_dir)

        # Fix 2: Auto-correct LCHARG for non-band-structure steps.
        # LLMs tend to set LCHARG=.FALSE. to save disk space, but preparatory steps
        # (relaxation, SCF) must write CHGCAR so downstream ICHARG=11 steps can read it.
        # Only override when this step is generating charge density (ICHARG < 10).
        try:
            icharg_val = int(str(incar_params.get("ICHARG", 2)))
        except (ValueError, TypeError):
            icharg_val = 2
        if icharg_val < 10:
            lcharg = incar_params.get("LCHARG", ".TRUE.")
            if lcharg in (False, ".FALSE.", "F", "False", "false"):
                logging.warning(
                    f"    Auto-correcting LCHARG: .FALSE. → .TRUE. for step '{step.name}' "
                    f"(ICHARG={icharg_val}; CHGCAR may be needed by downstream band structure steps)"
                )
                incar_params["LCHARG"] = ".TRUE."

        # Fix 4: Auto-resolve known VASP parameter conflicts
        self._resolve_param_conflicts(incar_params, step.name)

        # Write INCAR file
        incar_path = os.path.join(work_dir, "INCAR")
        with open(incar_path, 'w') as f:
            f.write("# INCAR generated by VASPExecutor\n")
            f.write(f"# Step: {step.name}\n")
            f.write(f"# {step.description}\n\n")
            
            for key, value in incar_params.items():
                # Format value appropriately
                if isinstance(value, bool):
                    value_str = ".TRUE." if value else ".FALSE."
                elif isinstance(value, list):
                    # Convert Python list to VASP space-separated format: [0,0,1] → "0 0 1"
                    value_str = " ".join(str(x) for x in value)
                elif isinstance(value, str):
                    value_str = value
                else:
                    value_str = str(value)

                f.write(f"{key} = {value_str}\n")
        
        logging.info(f"    Generated INCAR with {len(incar_params)} parameters")
    
    def _is_valid_incar_value(self, key: str, value) -> bool:
        """
        Check if an INCAR parameter value is valid.
        
        Filters out descriptive text that LLM might have added.
        
        Args:
            key: INCAR parameter name
            value: Parameter value
            
        Returns:
            True if value is valid, False otherwise
        """
        if value is None:
            return False
        
        # Convert to string for checking
        value_str = str(value).strip()
        
        # Check for empty values
        if not value_str:
            return False
        
        # VASP parameters that legitimately accept "Auto" as a value
        # (e.g., LREAL = Auto). These must NOT be filtered by the "auto"
        # pattern below, which targets LLM-generated descriptive text like
        # "auto-detect from POSCAR" or "automatically determined".
        vasp_auto_params = {"LREAL", "PRECFOCK", "METAGGA"}
        if key.upper() in vasp_auto_params and value_str.lower() == "auto":
            return True
        
        # Check for descriptive/invalid patterns
        invalid_patterns = [
            "set using",
            "auto",
            "see notes",
            "determine from",
            "evaluate",
            "recommended",
            "use factor",
            "calculate",
            "optional",
            "if needed",
            "based on",
            "refer to",
        ]
        
        value_lower = value_str.lower()
        for pattern in invalid_patterns:
            if pattern in value_lower:
                return False
        
        # Special handling for specific parameters
        if key == "NBANDS":
            # NBANDS must be an integer or omitted
            try:
                int(value)
                return True
            except (ValueError, TypeError):
                return False
        
        # Check for parameter names that look like notes/comments
        if key.endswith("_notes") or key.endswith("_note") or key.endswith("NOTE") or key.endswith("NOTES"):
            return False
        
        return True
    
    # =========================================================================
    # MAGMOM auto-detection for spin-polarized calculations
    # =========================================================================
    
    # Default initial MAGMOM values for common magnetic/transition-metal elements.
    # These are initial guesses to break spin symmetry; VASP will converge to
    # the correct magnetic state during SCF. Higher values for known magnetic
    # elements help convergence.
    MAGNETIC_ELEMENT_MAGMOM = {
        # 3d transition metals (most common magnetic elements in 2D materials)
        'Sc': 1.0, 'Ti': 2.0, 'V': 3.0, 'Cr': 5.0, 'Mn': 5.0,
        'Fe': 4.0, 'Co': 3.0, 'Ni': 2.0, 'Cu': 1.0,
        # 4d transition metals
        'Y': 1.0, 'Zr': 2.0, 'Nb': 3.0, 'Mo': 3.0, 'Tc': 3.0,
        'Ru': 2.0, 'Rh': 1.0, 'Pd': 0.6,
        # 5d transition metals
        'Hf': 2.0, 'Ta': 3.0, 'W': 3.0, 'Re': 3.0,
        'Os': 2.0, 'Ir': 1.0, 'Pt': 0.6,
        # Lanthanides (4f)
        'La': 0.6, 'Ce': 1.0, 'Pr': 3.0, 'Nd': 4.0, 'Pm': 5.0,
        'Sm': 5.0, 'Eu': 7.0, 'Gd': 7.0, 'Tb': 6.0, 'Dy': 5.0,
        'Ho': 4.0, 'Er': 3.0, 'Tm': 2.0, 'Yb': 1.0, 'Lu': 0.6,
        # Actinides (5f)
        'U': 3.0, 'Np': 3.0, 'Pu': 5.0,
    }
    DEFAULT_MAGMOM = 0.6  # Small non-zero value for symmetry breaking
    
    # Elements whose d-electrons require LMAXMIX >= 4
    D_BLOCK_ELEMENTS = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    }
    # Elements whose f-electrons require LMAXMIX >= 6
    F_BLOCK_ELEMENTS = {
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
        'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    }
    
    def _parse_poscar_elements(self, work_dir: str) -> list:
        """
        Parse POSCAR to get element types and counts.
        
        Args:
            work_dir: Directory containing POSCAR
            
        Returns:
            List of (element_symbol, count) tuples, e.g. [("Br", 2), ("Mn", 1)]
        """
        poscar_path = os.path.join(work_dir, "POSCAR")
        if not os.path.exists(poscar_path):
            return []
        
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 7:
                return []
            
            # Line 6 (0-indexed: 5) has element names
            # Line 7 (0-indexed: 6) has counts
            elements = lines[5].split()
            counts = [int(x) for x in lines[6].split()]
            
            if len(elements) != len(counts):
                logging.warning(f"    POSCAR element/count mismatch: {elements} vs {counts}")
                return []
            
            return list(zip(elements, counts))
        except Exception as e:
            logging.warning(f"    Failed to parse POSCAR elements: {e}")
            return []
    
    # Elements whose presence strongly indicates the material may be magnetic.
    # These are elements commonly found in FM/AFM 2D materials in C2DB.
    # Conservative set: only 3d transition metals with partially filled d-shells.
    MAGNETIC_TRIGGER_ELEMENTS = {'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu'}
    
    # Extended set: 4f/5f elements that are almost always magnetic.
    MAGNETIC_TRIGGER_RARE_EARTH = {
        'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
        'Ho', 'Er', 'Tm', 'Yb',
        'U', 'Np', 'Pu',
    }
    
    @staticmethod
    def _resolve_param_conflicts(incar_params: dict, step_name: str):
        """
        Auto-resolve known VASP parameter incompatibilities.

        Rules:
        - ADDGRID + IVDW=20/21 (Tkatchenko-Scheffler): VASP errors out.
          → Remove ADDGRID when TS dispersion is active.
        """
        ivdw = None
        try:
            ivdw = int(str(incar_params.get("IVDW", 0)))
        except (ValueError, TypeError):
            pass

        addgrid = str(incar_params.get("ADDGRID", ".FALSE.")).upper().strip()
        addgrid_on = addgrid in (".TRUE.", "TRUE", "T", ".T.")

        if addgrid_on and ivdw in (20, 21):
            logging.warning(
                f"    Parameter conflict in '{step_name}': ADDGRID=.TRUE. is incompatible "
                f"with IVDW={ivdw} (Tkatchenko-Scheffler). Removing ADDGRID."
            )
            del incar_params["ADDGRID"]

    def _auto_set_spin_magmom(self, incar_params: dict, work_dir: str):
        """
        Fill in spin-polarization defaults that the step parameters left unspecified.

        Explicit ISPIN/MAGMOM/LMAXMIX values from the step parameters are never
        overridden. Defaults are only added when they are missing:
        - ISPIN not set: ISPIN=2 if the composition contains magnetic trigger
          elements (V, Cr, Mn, Fe, Co, Ni, Cu, rare earths) or has an odd
          valence-electron count; otherwise VASP's default (ISPIN=1) is kept.
        - ISPIN=2 without MAGMOM: element-wise initial moments (collinear only).
        - ISPIN=2 without LMAXMIX: 4 for d-block, 6 for f-block elements.

        Args:
            incar_params: Dictionary of INCAR parameters (modified in-place)
            work_dir: Working directory containing POSCAR and POTCAR
        """
        elements = self._parse_poscar_elements(work_dir)
        if not elements:
            return
        all_element_names = {elem for elem, _ in elements}

        if "ISPIN" in incar_params:
            try:
                ispin = int(str(incar_params["ISPIN"]).strip())
            except ValueError:
                return
            if ispin != 2:
                return
        else:
            has_magnetic = bool(all_element_names & self.MAGNETIC_TRIGGER_ELEMENTS)
            has_rare_earth = bool(all_element_names & self.MAGNETIC_TRIGGER_RARE_EARTH)
            total_electrons = self._count_electrons_from_potcar(work_dir, elements)
            is_odd_electrons = total_electrons is not None and total_electrons % 2 == 1
            if not (has_magnetic or has_rare_earth or is_odd_electrons):
                return
            incar_params["ISPIN"] = 2
            logging.info(f"    ISPIN not specified; defaulting to ISPIN=2 "
                         f"(magnetic elements={has_magnetic or has_rare_earth}, odd electrons={is_odd_electrons})")

        noncollinear = any(
            str(incar_params.get(key, ".FALSE.")).upper().strip() in (".TRUE.", "TRUE", "T", ".T.")
            for key in ("LSORBIT", "LNONCOLLINEAR")
        )
        if "MAGMOM" not in incar_params and not noncollinear:
            magmom_parts = []
            for element, count in elements:
                mag = self.MAGNETIC_ELEMENT_MAGMOM.get(element, self.DEFAULT_MAGMOM)
                magmom_parts.append(f"{count}*{mag}")
            incar_params["MAGMOM"] = " ".join(magmom_parts)
            logging.info(f"    MAGMOM not specified; defaulting to MAGMOM = {incar_params['MAGMOM']}")

        if "LMAXMIX" not in incar_params:
            if all_element_names & self.F_BLOCK_ELEMENTS:
                incar_params["LMAXMIX"] = 6
            elif all_element_names & self.D_BLOCK_ELEMENTS:
                incar_params["LMAXMIX"] = 4

    def _count_electrons_from_potcar(
        self, work_dir: str, elements: list
    ) -> Optional[int]:
        """
        Count total valence electrons from POTCAR ZVAL entries.
        
        Args:
            work_dir: Directory containing POTCAR
            elements: List of (element, count) tuples from POSCAR
            
        Returns:
            Total electron count, or None if POTCAR not found/parseable
        """
        potcar_path = os.path.join(work_dir, "POTCAR")
        if not os.path.exists(potcar_path):
            return None
        
        try:
            zvals = []
            with open(potcar_path, 'r') as f:
                for line in f:
                    if 'ZVAL' in line and 'POMASS' in line:
                        # Format: "   POMASS =  107.868; ZVAL =  11.000    mass and valenz"
                        zval_part = line.split('ZVAL')[1]
                        # Extract the number after '='
                        zval_str = zval_part.split('=')[1].strip().split()[0]
                        zvals.append(float(zval_str))
            
            if len(zvals) != len(elements):
                logging.warning(f"    POTCAR ZVAL count ({len(zvals)}) != "
                                f"element count ({len(elements)})")
                return None
            
            total = sum(zval * count for zval, (_, count) in zip(zvals, elements))
            total_int = int(round(total))
            logging.info(f"    Total valence electrons from POTCAR: {total_int}")
            return total_int
            
        except Exception as e:
            logging.warning(f"    Failed to parse POTCAR ZVAL: {e}")
            return None
    
    def _parse_kpoint_grid(self, text: str) -> Optional[tuple[int, int, int]]:
        """
        Parse k-point grid from text description with robust pattern matching.
        
        Supports multiple formats:
        - "12x12x1", "12×12×1", "12*12*1", "12 X 12 X 1", "12 12 1"
        - "12x12x1 grid", "Monkhorst-Pack 12x12x1"
        - Handles various Unicode multiplication symbols
        
        Returns (nx, ny, nz) if found, None otherwise.
        """
        if not text:
            return None
        
        # Normalize text: replace various multiplication symbols with a standard separator
        # Handle: × (U+00D7), ✕ (U+2715), * (asterisk), X (capital X), x (lowercase x)
        normalized = text
        # Replace Unicode multiplication symbols with x
        normalized = normalized.replace('×', 'x')  # U+00D7 multiplication sign
        normalized = normalized.replace('✕', 'x')  # U+2715 multiplication X
        normalized = normalized.replace('*', 'x')   # Asterisk
        normalized = re.sub(r'\s*[Xx]\s*', 'x', normalized)  # Normalize spacing around x/X
        
        # Pattern 1: Standard format with separators (12x12x1, 12 x 12 x 1, etc.)
        # After normalization, all separators should be 'x'
        patterns = [
            r'(\d+)\s*x\s*(\d+)\s*x\s*(\d+)',  # With separators (normalized to x)
            r'(\d+)\s+(\d+)\s+(\d+)',  # Space-separated (only if 3 consecutive numbers)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                try:
                    nx, ny, nz = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    # Sanity check: reasonable k-point values
                    if 1 <= nx <= 1000 and 1 <= ny <= 1000 and 1 <= nz <= 1000:
                        return (nx, ny, nz)
                except (ValueError, IndexError):
                    continue
        
        # Pattern 2: Try to extract three consecutive numbers (more lenient)
        # This handles cases like "use 12 12 1 k-mesh" or "12, 12, 1"
        numbers = re.findall(r'\b(\d+)\b', normalized)
        if len(numbers) >= 3:
            # Try different combinations of the first 3-5 numbers
            for i in range(len(numbers) - 2):
                try:
                    nx, ny, nz = int(numbers[i]), int(numbers[i+1]), int(numbers[i+2])
                    if 1 <= nx <= 1000 and 1 <= ny <= 1000 and 1 <= nz <= 1000:
                        # Additional check: these should be in a reasonable context
                        # (not part of other numbers like dates, versions, etc.)
                        return (nx, ny, nz)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _prepare_kpoints(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        work_dir: str
    ):
        """
        Prepare KPOINTS file based on step specifications.
        
        step.KPOINTS can be:
        - A string description (e.g., "Dense Monkhorst-Pack 12x12x1")
        - A dict with vaspkit task specifications
        - Instructions to use KPATH.in
        
        This function uses multiple fallback strategies to ensure KPOINTS is generated:
        1. Try to parse explicit grid specification (with robust pattern matching)
        2. Try vaspkit with appropriate task
        3. Copy from previous step
        4. Generate default Gamma-only (last resort)
        """
        kpoints_path = os.path.join(work_dir, "KPOINTS")
        
        kpoints_spec = step.KPOINTS
        kpoints_generated = False
        
        # Check if this is a NEB calculation - use Gamma-only for molecules
        is_neb = self._is_neb_calculation(step)
        if is_neb:
            # NEB for molecules typically uses Gamma-only
            self._write_kpoints_gamma(work_dir)
            logging.info(f"    Generated Gamma-only KPOINTS for NEB")
            return
        
        if isinstance(kpoints_spec, str):
            # Parse string description
            if "k-path" in kpoints_spec.lower() or "high-symmetry" in kpoints_spec.lower():
                # Generate k-path using vaspkit directly
                # Use task 302 for 2D/slab, 303 for 3D bulk
                task_num = 302 if ("2d" in kpoints_spec.lower() or "slab" in kpoints_spec.lower()) else 303
                try:
                    run_command(f"vaspkit -task {task_num}", cwd=work_dir, shell=True)
                    kpath_in = os.path.join(work_dir, "KPATH.in")
                    if os.path.exists(kpath_in):
                        shutil.copy(kpath_in, kpoints_path)
                        logging.info(f"    Generated k-path using vaspkit -task {task_num}, copied KPATH.in → KPOINTS")
                        kpoints_generated = True
                        return
                    elif os.path.exists(kpoints_path):
                        logging.info(f"    vaspkit -task {task_num} generated KPOINTS directly")
                        kpoints_generated = True
                        return
                    else:
                        logging.warning(f"    vaspkit -task {task_num} ran but no KPOINTS/KPATH.in produced, falling through to fallback")
                except Exception as e:
                    logging.warning(f"    vaspkit -task {task_num} failed: {e}, falling through to fallback")
                # Fall through to fallback strategies below
            elif "gamma" in kpoints_spec.lower() and "only" in kpoints_spec.lower():
                # Gamma-only for molecules
                self._write_kpoints_gamma(work_dir)
                logging.info(f"    Generated Gamma-only KPOINTS")
                kpoints_generated = True
            elif "monkhorst" in kpoints_spec.lower() or "gamma" in kpoints_spec.lower() or "grid" in kpoints_spec.lower():
                # Try to extract grid using robust parser
                grid = self._parse_kpoint_grid(kpoints_spec)
                if grid:
                    nx, ny, nz = grid
                    self._write_kpoints_grid(work_dir, nx, ny, nz,
                                             gamma_centered="gamma" in str(kpoints_spec).lower())
                    logging.info(f"    Generated KPOINTS grid: {nx}x{ny}x{nz} (parsed from: {kpoints_spec[:50]})")
                    kpoints_generated = True
                else:
                    # Try vaspkit task 102 (automatic k-mesh generation)
                    logging.info(f"    KPOINTS: Could not parse grid from '{kpoints_spec[:50]}', trying vaspkit task 102")
                    try:
                        run_command(f"vaspkit -task 102", cwd=work_dir, shell=True)
                        if os.path.exists(kpoints_path):
                            logging.info(f"    Generated KPOINTS using vaspkit task 102")
                            kpoints_generated = True
                    except Exception as e:
                        logging.warning(f"    vaspkit task 102 failed: {e}")
            elif "density" in kpoints_spec.lower():
                # Use k-point density
                # Extract density value (e.g., "k-point density 6.0 per Å⁻¹")
                match = re.search(r'density\s+(\d+\.?\d*)', kpoints_spec)
                if match:
                    density = float(match.group(1))
                    # Convert GPAW/C2DB density (points per Å⁻¹) to vaspkit kpr (2π/Å spacing)
                    # GPAW: N = density * 2π / |a|
                    # vaspkit: N = 1 / (|a| * kpr)
                    # → kpr = 1 / (2π * density)
                    import math
                    kpr = math.floor(1.0 / (2.0 * math.pi * density) * 1000) / 1000
                    try:
                        run_command(f"vaspkit -task 102 -kpr {kpr:.6f}", cwd=work_dir, shell=True)
                        if os.path.exists(kpoints_path):
                            logging.info(f"    Generated KPOINTS: C2DB density {density}/Å⁻¹ → vaspkit kpr={kpr:.4f}")
                            kpoints_generated = True
                    except Exception as e:
                        logging.warning(f"    vaspkit density generation failed: {e}")
                else:
                    logging.warning(f"    KPOINTS: 'density' keyword found but could not parse numeric value from '{kpoints_spec[:60]}'")
                    logging.warning(f"    Falling back to vaspkit task 102 with default density")
                    try:
                        run_command(f"vaspkit -task 102", cwd=work_dir, shell=True)
                        if os.path.exists(kpoints_path):
                            logging.info(f"    Generated KPOINTS using vaspkit task 102 (density parse fallback)")
                            kpoints_generated = True
                    except Exception as e:
                        logging.warning(f"    vaspkit fallback failed: {e}")
            else:
                # Generic case: try to extract grid from any string
                grid = self._parse_kpoint_grid(kpoints_spec)
                if grid:
                    nx, ny, nz = grid
                    self._write_kpoints_grid(work_dir, nx, ny, nz,
                                             gamma_centered="gamma" in str(kpoints_spec).lower())
                    logging.info(f"    Generated KPOINTS grid: {nx}x{ny}x{nz} (parsed from: {kpoints_spec[:50]})")
                    kpoints_generated = True
                else:
                    # Try vaspkit as fallback
                    logging.info(f"    KPOINTS: Could not parse grid from '{kpoints_spec[:50]}', trying vaspkit task 102")
                    try:
                        run_command(f"vaspkit -task 102", cwd=work_dir, shell=True)
                        if os.path.exists(kpoints_path):
                            logging.info(f"    Generated KPOINTS using vaspkit task 102")
                            kpoints_generated = True
                    except Exception as e:
                        logging.warning(f"    vaspkit task 102 failed: {e}")
        
        elif isinstance(kpoints_spec, dict):
            # Use vaspkit with specified options
            if "vaspkit_task" in kpoints_spec:
                task = int(kpoints_spec["vaspkit_task"])
                kpr = kpoints_spec.get("kpr", kpoints_spec.get("options", None))
                try:
                    if kpr:
                        run_command(f"vaspkit -task {task} -kpr {kpr}", cwd=work_dir, shell=True)
                    else:
                        run_command(f"vaspkit -task {task}", cwd=work_dir, shell=True)
                    if os.path.exists(kpoints_path):
                        logging.info(f"    Generated KPOINTS using vaspkit task {task}")
                        kpoints_generated = True
                except Exception as e:
                    logging.warning(f"    vaspkit task {task} failed: {e}")
        
        # If no KPOINTS was generated and none exists, use fallback strategies
        if not os.path.exists(kpoints_path):
            # Strategy 1: Try to copy from previous VASP step
            for step_dir in reversed(list(context.vasp_completed_steps.values())):
                prev_kpoints = os.path.join(step_dir, "KPOINTS")
                if os.path.exists(prev_kpoints):
                    shutil.copy(prev_kpoints, kpoints_path)
                    logging.info(f"    Copied KPOINTS from previous VASP step: {step_dir}")
                    return
            
            # Strategy 2: Try vaspkit one more time if we haven't tried it yet
            if not kpoints_generated:
                logging.info(f"    KPOINTS: Trying vaspkit task 102 as fallback")
                try:
                    run_command(f"vaspkit -task 102", cwd=work_dir, shell=True)
                    if os.path.exists(kpoints_path):
                        logging.info(f"    Generated KPOINTS using vaspkit task 102 (fallback)")
                        return
                except Exception as e:
                    logging.warning(f"    vaspkit fallback failed: {e}")
            
            # Strategy 3: Generate default Gamma-only (last resort)
            logging.warning(f"    KPOINTS: All strategies failed, generating default Gamma-only KPOINTS")
            logging.warning(f"    Original KPOINTS spec was: {kpoints_spec}")
            self._write_kpoints_gamma(work_dir)
            logging.info(f"    Generated default Gamma-only KPOINTS")
    
    def _write_kpoints_grid(self, work_dir: str, nx: int, ny: int, nz: int, gamma_centered: bool = False):
        """Write an automatic KPOINTS mesh (Monkhorst-Pack, or Gamma-centered if requested)."""
        kpoints_path = os.path.join(work_dir, "KPOINTS")
        with open(kpoints_path, 'w') as f:
            f.write("Automatic mesh\n")
            f.write("0\n")
            f.write("Gamma\n" if gamma_centered else "Monkhorst-Pack\n")
            f.write(f"{nx} {ny} {nz}\n")
            f.write("0 0 0\n")
    
    def _write_kpoints_gamma(self, work_dir: str):
        """Write Gamma-only KPOINTS file (for molecules/NEB)."""
        kpoints_path = os.path.join(work_dir, "KPOINTS")
        with open(kpoints_path, 'w') as f:
            f.write("Gamma-point only\n")
            f.write("0\n")
            f.write("Gamma\n")
            f.write("1 1 1\n")
            f.write("0 0 0\n")
    
    def _copy_additional_files(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        work_dir: str
    ):
        """
        Copy additional required files (CHGCAR, WAVECAR, WAVEDER, etc.).
        
        Supports:
        - Standard files: CHGCAR, WAVECAR, WAVEDER
        - GW files: WAVECAR, WAVEDER for GW calculations
        - SOC files: WAVECAR, CHGCAR from collinear
        - NEB files: Multiple POSCAR images
        - Constrained MD: ICONST
        - Dimer: MODECAR
        - MLFF: ML_AB, ML_FF
        """
        # Extended file list for advanced calculations
        standard_files = ["CHGCAR", "WAVECAR", "WAVEDER", "LOCPOT", "DYNMAT"]
        special_files = ["ICONST", "MODECAR", "ML_AB", "ML_FF"]
        
        all_files = standard_files + special_files
        
        # Note: per-step file requirements (CHGCAR, WAVECAR, WAVEDER, ...) must be
        # declared explicitly by the step planner via step.input_files_needed.
        # Do NOT auto-append based on step-name substrings ("soc", "gw", ...) —
        # that pattern historically caused false positives (e.g. a relaxation step
        # named "relax (no SOC)" was treated as a SOC step and demanded CHGCAR on
        # the first VASP call). The only rule-based augmentation we still apply
        # is the physics-derived "ICHARG>=10 requires CHGCAR", handled earlier in
        # execute_step() before this helper runs.

        # Copy files from appropriate source steps
        for filename in all_files:
            if filename not in step.input_files_needed:
                continue
            
            # Search in completed steps (reverse order to find most recent)
            source_found = False
            
            # First check explicitly declared dependencies
            for dep_step_name in step.depends_on:
                if dep_step_name in context.completed_steps:
                    dep_dir = context.completed_steps[dep_step_name]
                    source_path = os.path.join(dep_dir, filename)
                    if os.path.exists(source_path):
                        self._copy_or_link_file(source_path, work_dir, filename)
                        source_found = True
                        break
            
            # Fall back to most recent VASP step
            if not source_found and context.vasp_completed_steps:
                for step_dir in reversed(list(context.vasp_completed_steps.values())):
                    source_path = os.path.join(step_dir, filename)
                    if os.path.exists(source_path):
                        self._copy_or_link_file(source_path, work_dir, filename)
                        source_found = True
                        break
            
            if not source_found and filename in step.input_files_needed:
                logging.warning(f"    Required file {filename} not found in previous steps")
    
    def _copy_or_link_file(self, source_path: str, work_dir: str, filename: str):
        """
        Copy or symlink a file based on its size.
        
        Large files (WAVECAR, WAVEDER) are symlinked to save space.
        """
        # Files that should be symlinked (> 100 MB typically)
        large_files = ["WAVECAR", "WAVEDER", "ML_AB", "ML_FF"]
        
        target_path = os.path.join(work_dir, filename)
        
        if filename in large_files:
            # Use symlink for large files
            if os.path.exists(target_path):
                os.remove(target_path)
            try:
                # Use realpath to resolve any existing symlink chain,
                # preventing "Too many levels of symbolic links" (ELOOP)
                real_source = os.path.realpath(source_path)
                os.symlink(real_source, target_path)
                logging.info(f"    Symlinked {filename} from previous step")
            except OSError:
                # Fallback to copy if symlink fails
                shutil.copy(source_path, target_path)
                logging.info(f"    Copied {filename} (symlink failed)")
        else:
            shutil.copy(source_path, target_path)
            logging.info(f"    Copied {filename} from previous step")
    
    def _execute_vaspkit_actions(self, step: WorkflowStep, work_dir: str):
        """
        Execute VASPKIT actions specified in the step.
        
        Args:
            step: WorkflowStep with vaspkit_actions list
            work_dir: Working directory
        """
        if not step.vaspkit_actions:
            return
        
        logging.info(f"  Executing {len(step.vaspkit_actions)} VASPKIT actions")
        
        for action in step.vaspkit_actions:
            try:
                action_lower = action.lower()
                
                # Helper: extract k-point density from action string and convert
                # to vaspkit KSPACING.
                # Handles: "density 6.0/Å⁻¹", "density 12.0 per Å⁻¹", etc.
                # Returns (density, kspacing) tuple, or (None, None) if not found.
                def _extract_density_and_kspacing(text):
                    import math
                    m = re.search(r'density\s+(?:of\s+)?(\d+\.?\d*)', text, re.IGNORECASE)
                    if not m:
                        m = re.search(r'(\d+\.?\d*)\s*(?:per\s+)?[AÅ/]', text)
                    if m:
                        density = float(m.group(1))
                        # Convert C2DB/GPAW density (per Å⁻¹) to vaspkit KSPACING (2π/Å):
                        # KSPACING = 1 / (2π × density)
                        kspacing = math.floor(1.0 / (2.0 * math.pi * density) * 1000) / 1000
                        return density, kspacing
                    return None, None
                
                # Parse action description and execute
                if "generate" in action_lower and "k-path" in action_lower:
                    # Generate k-path using command-line mode
                    if "2d" in action_lower or "302" in action:
                        task_num = 302
                    elif "1d" in action_lower or "301" in action:
                        task_num = 301
                    else:
                        task_num = 303
                    
                    # Extract k-point density if specified (e.g., "with k-point density 12.0/Å⁻¹")
                    density, kspacing = _extract_density_and_kspacing(action)
                    if kspacing is not None:
                        run_command(f"vaspkit -task {task_num} -kpr {kspacing:.6f}", cwd=work_dir, shell=True)
                        logging.info(f"    vaspkit -task {task_num} -kpr {kspacing:.5f} (density={density}/Å⁻¹)")
                    else:
                        run_command(f"vaspkit -task {task_num}", cwd=work_dir, shell=True)
                        logging.info(f"    vaspkit -task {task_num} (default density)")
                    
                    # Copy KPATH.in to KPOINTS
                    kpath_in = os.path.join(work_dir, "KPATH.in")
                    if os.path.exists(kpath_in):
                        shutil.copy(kpath_in, os.path.join(work_dir, "KPOINTS"))
                        logging.info(f"    Generated k-path and copied to KPOINTS")
                
                elif "task" in action_lower and re.search(r'task\s+(\d+)', action_lower):
                    # Execute a specific vaspkit task (e.g., "Use vaspkit task 102 with k-point density 6.0/Å⁻¹")
                    task_match = re.search(r'task\s+(\d+)', action_lower)
                    task_num = int(task_match.group(1))
                    density, kspacing = _extract_density_and_kspacing(action)
                    
                    if kspacing is not None:
                        run_command(f"vaspkit -task {task_num} -kpr {kspacing:.6f}", cwd=work_dir, shell=True)
                        logging.info(f"    Executed vaspkit -task {task_num} -kpr {kspacing:.5f} (density={density}/Å⁻¹)")
                    else:
                        run_command(f"vaspkit -task {task_num}", cwd=work_dir, shell=True)
                        logging.info(f"    Executed vaspkit -task {task_num}")
                    
                    # If task 102 generated KPOINTS, note it
                    if task_num == 102 and os.path.exists(os.path.join(work_dir, "KPOINTS")):
                        logging.info(f"    KPOINTS generated by vaspkit task 102")
                
                elif ("extract" in action_lower or "compute" in action_lower) and ("vbm" in action_lower or "cbm" in action_lower or "band" in action_lower):
                    # Band structure extraction and VBM/CBM detection
                    # These are typically post-processing, so just log
                    logging.info(f"    VASPKIT action: {action}")
                
                elif "dos" in action_lower and ("plot" in action_lower or "extract" in action_lower):
                    # DOS plotting/extraction
                    logging.info(f"    VASPKIT action: {action}")
                
                elif "check" in action_lower or "inspect" in action_lower:
                    # Inspection actions - informational only
                    logging.debug(f"    Inspection action: {action}")
                
                elif "no vaspkit" in action_lower or "not required" in action_lower or not action.strip():
                    # Explicitly no action needed or empty string
                    logging.debug(f"    No VASPKIT action needed: {action}")
                
                else:
                    # Unknown action - log for information
                    logging.info(f"    VASPKIT action (informational): {action}")
            
            except Exception as e:
                logging.warning(f"    VASPKIT action failed (non-critical): {action}, error: {e}")
    
    def _validate_outputs(self, step: WorkflowStep, work_dir: str) -> List[str]:
        """
        Validate that expected output files were produced.
        
        Args:
            step: WorkflowStep definition
            work_dir: Working directory
            
        Returns:
            List of files actually produced
            
        Raises:
            FileNotFoundError: If critical output files are missing
        """
        outputs_produced = []
        missing_files = []
        empty_files = []
        
        # Check if this is a NEB calculation - outputs are in subdirectories
        is_neb = self._is_neb_calculation(step)
        
        for expected_file in step.output_files_produced:
            # Clean the filename by removing parenthetical descriptions if present
            # E.g., "CONTCAR (relaxed structure)" -> "CONTCAR"
            clean_filename = expected_file.split('(')[0].strip()
            
            file_path = os.path.join(work_dir, clean_filename)
            
            # For NEB calculations, check subdirectories for output files
            if is_neb and not os.path.exists(file_path):
                # Check in image subdirectories (01/, 02/, etc.)
                found_in_subdir = False
                for subdir in os.listdir(work_dir):
                    subdir_path = os.path.join(work_dir, subdir)
                    if os.path.isdir(subdir_path) and subdir.isdigit():
                        subdir_file = os.path.join(subdir_path, clean_filename)
                        if os.path.exists(subdir_file) and os.path.getsize(subdir_file) > 0:
                            found_in_subdir = True
                            break
                
                if found_in_subdir:
                    outputs_produced.append(f"{clean_filename} (in image dirs)")
                    continue
            
            if not os.path.exists(file_path):
                missing_files.append(clean_filename)
            elif os.path.getsize(file_path) == 0:
                empty_files.append(clean_filename)
            else:
                outputs_produced.append(clean_filename)
        
        # Check for critical failures
        critical_files = ["CONTCAR", "OUTCAR", "vasprun.xml"]

        # If LCHARG is not explicitly .FALSE., CHGCAR should be non-empty after a VASP run.
        # An empty CHGCAR indicates VASP crashed before writing charge density (e.g. ZBRENT, EDDDAV).
        incar_params = getattr(step, 'INCAR', None) or {}
        lcharg = str(incar_params.get("LCHARG", ".TRUE.")).upper().strip()
        if lcharg not in {".FALSE.", "FALSE", "F"}:
            chgcar_path = os.path.join(work_dir, "CHGCAR")
            if os.path.exists(chgcar_path) and os.path.getsize(chgcar_path) == 0:
                error_detail = self._extract_vasp_error(work_dir)
                error_msg = "CHGCAR is empty — VASP likely crashed before writing charge density"
                if error_detail:
                    error_msg += f". VASP error: {error_detail}"
                raise RuntimeError(error_msg)

        # For NEB calculations, we only need files in subdirectories
        if is_neb:
            # Check if at least one image directory has the critical files
            neb_success = False
            for subdir in os.listdir(work_dir):
                subdir_path = os.path.join(work_dir, subdir)
                if os.path.isdir(subdir_path) and subdir.isdigit() and subdir != "00":
                    # Skip endpoint directories (00 and last), check intermediate images
                    outcar_path = os.path.join(subdir_path, "OUTCAR")
                    if os.path.exists(outcar_path) and os.path.getsize(outcar_path) > 0:
                        neb_success = True
                        break
            
            if neb_success:
                logging.info(f"  NEB calculation completed - outputs in image subdirectories")
                return outputs_produced
        
        # Build detailed error message
        error_parts = []
        missing_critical = [f for f in missing_files if f in critical_files]
        empty_critical = [f for f in empty_files if f in critical_files]
        
        if missing_critical:
            error_parts.append(f"missing: {missing_critical}")
        if empty_critical:
            error_parts.append(f"empty: {empty_critical}")
        
        if error_parts:
            # Check OUTCAR for error details
            error_detail = self._extract_vasp_error(work_dir)
            error_msg = f"Critical output files {', '.join(error_parts)}"
            if error_detail:
                error_msg += f". VASP error: {error_detail}"
            raise FileNotFoundError(error_msg)
        
        # Log non-critical missing files
        if missing_files:
            logging.warning(f"  Some expected files not produced: {missing_files}")
        if empty_files:
            logging.warning(f"  Some expected files are empty: {empty_files}")
        
        return outputs_produced
    
    def _extract_vasp_error(self, work_dir: str) -> str:
        """Extract error message from VASP output files."""
        outcar_path = os.path.join(work_dir, "OUTCAR")
        if os.path.exists(outcar_path):
            try:
                with open(outcar_path, 'r') as f:
                    content = f.read()
                
                # Look for error messages
                if "VERY BAD NEWS" in content or "ERROR" in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if "ERROR" in line or "VERY BAD NEWS" in line:
                            # Get a few lines of context
                            error_lines = lines[max(0, i-1):min(len(lines), i+5)]
                            return ' '.join(error_lines).strip()
                
                # Check for other failure indicators
                if "I REFUSE TO CONTINUE" in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if "Error reading" in line or "REFUSE" in line:
                            error_lines = lines[max(0, i-2):min(len(lines), i+3)]
                            return ' '.join([l.strip() for l in error_lines if l.strip()])
            except Exception:
                pass
        
        return ""
    
    def _perform_sanity_checks(self, step: WorkflowStep, work_dir: str) -> List[str]:
        """
        Perform sanity checks defined in the step.
        
        Args:
            step: WorkflowStep with sanity_checks list
            work_dir: Working directory
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        if not step.sanity_checks:
            return warnings
        
        logging.info(f"  Performing {len(step.sanity_checks)} sanity checks")
        
        outcar_path = os.path.join(work_dir, "OUTCAR")
        
        for check in step.sanity_checks:
            try:
                if "force" in check.lower():
                    # Check forces
                    if os.path.exists(outcar_path):
                        max_force = self._get_max_force(outcar_path)
                        if max_force is not None:
                            # Extract threshold from check description
                            match = re.search(r'(\d+\.?\d*)\s*eV', check)
                            threshold = float(match.group(1)) if match else 0.05
                            
                            if max_force > threshold:
                                warnings.append(
                                    f"Max force {max_force:.3f} eV/Å exceeds threshold {threshold}"
                                )
                
                elif "convergen" in check.lower():
                    # Check electronic convergence
                    if os.path.exists(outcar_path):
                        converged = self._check_convergence(outcar_path)
                        if not converged:
                            warnings.append("Electronic convergence not achieved")
                
                elif "nsw" in check.lower():
                    # Check if ionic steps hit NSW limit
                    if os.path.exists(outcar_path):
                        hit_nsw = self._check_nsw_limit(outcar_path)
                        if hit_nsw:
                            warnings.append("Ionic steps hit NSW limit - consider increasing")
                
                else:
                    logging.debug(f"    Check: {check}")
            
            except Exception as e:
                logging.warning(f"    Sanity check failed: {check}, error: {e}")
        
        if warnings:
            logging.warning(f"  Sanity check warnings: {warnings}")
        else:
            logging.info(f"  All sanity checks passed")
        
        return warnings
    
    def _get_max_force(self, outcar_path: str) -> Optional[float]:
        """Extract maximum force from OUTCAR."""
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
            
            # Find all force lines
            import re
            forces = re.findall(r'TOTAL-FORCE.*?\n(.*?)\n.*?-{10}', content, re.DOTALL)
            if forces:
                last_forces = forces[-1]
                # Parse force magnitudes
                max_force = 0.0
                for line in last_forces.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 6:
                        fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                        force_mag = (fx**2 + fy**2 + fz**2)**0.5
                        max_force = max(max_force, force_mag)
                return max_force
        except Exception as e:
            logging.debug(f"Could not extract forces: {e}")
        return None
    
    def _check_convergence(self, outcar_path: str) -> bool:
        """Check if electronic convergence was achieved."""
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
            
            # Look for convergence indicators
            if "reached required accuracy" in content:
                return True
            if "aborting loop" in content.lower():
                return False
            
            return True  # Assume converged if no abort message
        except Exception:
            return True
    
    def _check_nsw_limit(self, outcar_path: str) -> bool:
        """Check if ionic steps hit NSW limit."""
        try:
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
            
            # Find NSW setting and actual steps
            nsw = None
            actual_steps = 0
            
            for line in lines:
                if "NSW" in line and "=" in line:
                    parts = line.split("=")
                    if len(parts) > 1:
                        try:
                            nsw = int(parts[1].split()[0])
                        except ValueError:
                            pass
                
                if "Iteration" in line or "free  energy" in line:
                    actual_steps += 1
            
            if nsw and actual_steps >= nsw - 1:
                return True
        except Exception:
            pass
        return False
    
    def _extract_data(self, step: WorkflowStep, work_dir: str) -> Dict[str, Any]:
        """
        Extract data from output files (for postprocessing steps).
        
        Args:
            step: WorkflowStep definition
            work_dir: Working directory
            
        Returns:
            Dictionary of extracted data
        """
        # This would be implemented based on extraction_rules
        # For now, return empty dict
        return {}

    def execute(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        step_index: int,
        timeout_hours: Optional[float] = None,
        enable_monitoring: bool = True,
    ) -> StepExecutionResult:
        return self.execute_step(
            step=step,
            context=context,
            step_index=step_index,
            timeout_hours=timeout_hours,
            enable_monitoring=enable_monitoring,
        )

    def generate_step_outcome_summary(
        self,
        step: WorkflowStep,
        execution_result: StepExecutionResult,
        context: ExecutionContext,
    ) -> StepOutcomeSummary:
        output_dir = execution_result.output_dir
        outcar_path = os.path.join(output_dir, "OUTCAR")
        oszicar_path = os.path.join(output_dir, "OSZICAR")
        eigenval_path = os.path.join(output_dir, "EIGENVAL")
        contcar_path = os.path.join(output_dir, "CONTCAR")
        poscar_before = os.path.join(output_dir, "POSCAR")

        symmetry = None
        if os.path.exists(contcar_path):
            symmetry_info = extract_space_group(contcar_path)
            symmetry = symmetry_info.get("space_group_symbol")

        electronic_steps = extract_electronic_convergence(oszicar_path)

        # Compute cell volume change percentage
        vol_before = extract_cell_volume(poscar_before)
        vol_after = extract_cell_volume(contcar_path)
        vol_change_pct = None
        if vol_before and vol_after and vol_before > 0:
            vol_change_pct = (vol_after - vol_before) / vol_before * 100.0

        return StepOutcomeSummary(
            step_name=step.name,
            converged=execution_result.status == "success",
            total_energy=extract_total_energy(outcar_path) if os.path.exists(outcar_path) else None,
            ionic_steps_count=len(electronic_steps),
            electronic_steps_per_ionic=electronic_steps,
            max_force=extract_max_force(outcar_path),
            stress_tensor=extract_stress_tensor(outcar_path),
            symmetry_after=symmetry,
            cell_volume_change_pct=vol_change_pct,
            bandgap_estimate=estimate_bandgap_from_eigenval(eigenval_path),
            warnings=list(execution_result.warnings),
            raw_oszicar_tail=extract_oszicar_tail(oszicar_path),
            raw_outcar_summary=extract_outcar_summary(outcar_path),
            timing=dict(execution_result.metrics),
            metadata={
                "output_dir": output_dir,
                "symmetry_change": detect_symmetry_change(poscar_before, contcar_path)
                if os.path.exists(poscar_before) and os.path.exists(contcar_path)
                else None,
            },
        )
