"""
Job management for VASP calculations with Slurm and Local execution support.

This module handles all subprocess execution, job submission (Slurm or Local),
and job monitoring with integrated calculation assessment.

Execution modes:
- "slurm": Use Slurm job scheduler
- "local": Run VASP directly as a local process
- "auto": Auto-detect Slurm availability, fall back to local
"""
import subprocess
import logging
import time
import os
import signal
import shutil
from typing import Optional, Dict, Any

from config import (
    VASP_STD_CMD,
    VASP_NCL_CMD,
    VASP_GAM_CMD,
    VASP_ENV_SCRIPT,
    EXECUTION_MODE,
    LOCAL_MPI_PROCESSES,
    LOCAL_OMP_THREADS,
    GPU_ID,
    SLURM_HEADER,
    MONITOR_CHECK_INTERVAL,
)


# ============================================================
# VASP VERSION SELECTION
# ============================================================

def detect_vasp_version(incar_path: str) -> str:
    """
    Detect which VASP version should be used based on INCAR settings.
    
    Args:
        incar_path: Path to the INCAR file
        
    Returns:
        One of: "std", "ncl", "gam"
    """
    if not os.path.exists(incar_path):
        return "std"
    
    try:
        with open(incar_path, 'r') as f:
            content = f.read().upper()
        
        # Check for non-collinear / SOC keywords
        # These require vasp_ncl
        ncl_keywords = [
            'LNONCOLLINEAR',  # Non-collinear magnetism
            'LSORBIT',        # Spin-orbit coupling
        ]
        
        for keyword in ncl_keywords:
            # Match "KEYWORD = .TRUE." or "KEYWORD=.TRUE." or "KEYWORD=T" etc.
            import re
            pattern = rf'{keyword}\s*=\s*\.?T'
            if re.search(pattern, content):
                return "ncl"
        
        # Check for gamma-only (KPOINTS with only Gamma point)
        # This is optional optimization, skip for now
        
        return "std"
    except Exception:
        return "std"


def get_vasp_command(version: str = "std") -> str:
    """
    Get the VASP command for the specified version.
    
    Args:
        version: One of "std", "ncl", "gam"
        
    Returns:
        Full path to VASP executable
    """
    if version == "ncl":
        return VASP_NCL_CMD
    elif version == "gam":
        return VASP_GAM_CMD
    else:
        return VASP_STD_CMD


# ============================================================
# EXECUTION MODE DETECTION
# ============================================================

def is_slurm_available() -> bool:
    """
    Check if Slurm is available on the system.
    
    Returns:
        True if sbatch command is found and functional, False otherwise.
    """
    try:
        result = subprocess.run(
            ['which', 'sbatch'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return False
        
        # Also verify squeue works (sbatch might exist but slurm not running)
        result = subprocess.run(
            ['squeue', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def get_execution_mode() -> str:
    """
    Determine the actual execution mode based on configuration and availability.

    Returns:
        "slurm" or "local"
    """
    mode = EXECUTION_MODE.lower()

    if mode == "slurm":
        if not is_slurm_available():
            logging.error("EXECUTION_MODE is 'slurm' but Slurm is not available!")
            raise RuntimeError("Slurm is not available on this system")
        return "slurm"
    
    elif mode == "local":
        return "local"
    
    else:  # "auto" or any other value
        if is_slurm_available():
            logging.info("Auto-detected Slurm availability: using Slurm execution")
            return "slurm"
        else:
            logging.info("Slurm not available: using local execution mode")
            return "local"


# ============================================================
# COMMON UTILITIES
# ============================================================

def run_command(command: str, cwd: str = ".", shell: bool = False, sbatch: bool = False) -> str:
    """
    Execute a shell command and return its output.
    
    Args:
        command: Command to execute
        cwd: Working directory for the command
        shell: Whether to use shell execution
        sbatch: Whether to submit via sbatch (only for Slurm mode)
        
    Returns:
        Standard output from the command
        
    Raises:
        subprocess.CalledProcessError: If command fails
    """
    import sys
    
    logging.info(f"Executing in {cwd}: {command}")
    try:
        # Prepare environment - ensure Python is available for vaspkit
        env = os.environ.copy()
        if 'vaspkit' in command:
            # Set PYTHON_BIN for vaspkit to find Python in conda environments
            python_path = sys.executable
            env['PYTHON_BIN'] = python_path
            python_dir = os.path.dirname(python_path)
            if python_dir not in env.get('PATH', ''):
                env['PATH'] = f"{python_dir}:{env.get('PATH', '')}"
        
        # When using pipes or heredoc (e.g., for vaspkit), shell=True is required
        if "|" in command or "<<" in command:
            shell = True
            # Use module load only if configured (empty string means no module load needed)
            from config import VASPKIT_MODULE_LOAD
            if VASPKIT_MODULE_LOAD:
                full_command = f"{VASPKIT_MODULE_LOAD}; {command}"
            else:
                full_command = command
        else:
            if sbatch:
                full_command = [
                    'sbatch',
                    command
                ]
            else:
                full_command = command
        
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
            shell=shell,
            executable='/bin/bash' if shell else None,
            encoding='utf-8',
            env=env
        )
        
        logging.debug(result.stdout)
        if result.stderr:
            logging.warning(f"STDERR: {result.stderr}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed in {cwd}: {full_command}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise


# ============================================================
# SLURM EXECUTION MODE
# ============================================================

def generate_slurm_script(job_name: str, cwd: str, vasp_version: str = None) -> str:
    """
    Generate a Slurm submission script for VASP job.
    
    Args:
        job_name: Name for the Slurm job
        cwd: Directory where script will be created
        vasp_version: VASP version to use ("std", "ncl", "gam", or None for auto-detect)
        
    Returns:
        Path to the generated script
    """
    # Auto-detect VASP version from INCAR if not specified
    if vasp_version is None:
        incar_path = os.path.join(cwd, "INCAR")
        vasp_version = detect_vasp_version(incar_path)
        logging.info(f"Auto-detected VASP version for Slurm: {vasp_version}")
    
    # Get the appropriate VASP executable
    vasp_executable = get_vasp_command(vasp_version)
    
    script_path = os.path.join(cwd, "submit_vasp.sh")
    env_setup = f'source "{VASP_ENV_SCRIPT}"' if VASP_ENV_SCRIPT else "# No VASP environment script configured"
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=vasp_run.out
#SBATCH --error=vasp_run.err
{SLURM_HEADER}

{env_setup}
export OMP_NUM_THREADS={LOCAL_OMP_THREADS}

# Run VASP calculation (version: {vasp_version})
mpirun -np {max(1, LOCAL_MPI_PROCESSES)} {vasp_executable} > print-out
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path


def submit_slurm_job(script_path: str, cwd: str) -> str:
    """
    Submit a Slurm job and return the job ID.
    
    Args:
        script_path: Path to the Slurm script
        cwd: Working directory
        
    Returns:
        Slurm job ID
        
    Raises:
        IndexError: If job ID cannot be parsed from sbatch output
    """
    stdout = run_command(script_path, cwd=cwd, sbatch=True)
    try:
        job_id = stdout.split()[-1]
        logging.info(f"Slurm job submitted: {job_id} for {os.path.basename(cwd)}")
        return job_id
    except IndexError:
        logging.error(f"Could not parse job ID from sbatch output: {stdout}")
        raise


def wait_for_slurm_job(
    job_id: str, 
    poll_interval: int = 60,
    timeout_hours: float = None,
    calc_dir: str = None,
    enable_monitoring: bool = True,
    monitor_interval: int = MONITOR_CHECK_INTERVAL,
    monitor_config: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
    physical_context: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Poll Slurm queue and wait for job to complete.
    
    This function blocks until the specified job finishes.
    Tracks pending vs running time separately, and only counts running time
    against the timeout limit.
    
    Args:
        job_id: Slurm job ID to monitor
        poll_interval: Time in seconds between status checks (default: 60)
        timeout_hours: Maximum RUNNING time in hours (excluding pending time). 
                      None means no timeout.
        calc_dir: Directory containing the calculation (required for monitoring)
        enable_monitoring: Whether to enable rule-monitor checks (default: True)
        monitor_interval: Interval between rule-based monitoring checks in seconds (default: 600)
    
    Returns:
        dict with keys:
            - pending_time: Time spent in pending state (seconds)
            - running_time: Time spent in running state (seconds)  
            - total_time: Total time (pending + running) in seconds
            - timed_out: Whether the job was cancelled due to timeout
            - terminated_by_rule_monitor: Whether the rule monitor requested termination
            - monitor_assessment: Final assessment from the monitor stack (if enabled)
    """
    logging.info(f"Waiting for Slurm job {job_id} to complete...")
    if timeout_hours:
        logging.info(f"Timeout limit: {timeout_hours} hours (running time only, excluding pending)")
    
    # Initialize monitoring agent if enabled and calc_dir is provided
    monitor = None
    if enable_monitoring and calc_dir:
        try:
            from agents.reasoning_monitor import ReasoningMonitor

            # A fresh monitor per job (rule-monitor milestone state is per job), sharing
            # the workflow's LLM client so monitor calls count toward the token budget.
            monitor = ReasoningMonitor(config=monitor_config, llm_client=llm_client)
            logging.info(f"ReasoningMonitor enabled: checking OSZICAR/OUTCAR every {monitor_interval}s")
        except ImportError as e:
            logging.warning(f"Could not import ReasoningMonitor: {e}. Monitoring disabled.")
            monitor = None
    
    submit_time = time.time()
    running_start_time = None
    timeout_start_time = None  # Separate timer for timeout (can be reset by the monitor stack)
    pending_time = 0
    running_time = 0
    timed_out = False
    terminated_by_rule_monitor = False
    last_monitor_check = None
    monitor_assessment = None
    timeout_reset_count = 0  # Track how many times timeout was reset
    monitor_check_count = 0  # Number of rule-based monitor checks so far
    
    while True:
        try:
            # Check if job is still running
            stdout = run_command(f"squeue -j {job_id} -h -o %t", shell=True)
            if not stdout.strip():
                logging.info(f"Job {job_id} is no longer in queue. Assuming complete.")
                break
            
            job_state = stdout.strip()
            current_time = time.time()
            
            # Track when job starts running (transitions from PD to R)
            if job_state == "R" and running_start_time is None:
                running_start_time = current_time
                timeout_start_time = current_time  # Initialize timeout timer
                pending_time = current_time - submit_time
                logging.info(f"Job {job_id} started running after {pending_time/60:.1f} min pending")
            
            # Check timeout (based on time since last reset, not total running time)
            if timeout_start_time is not None and timeout_hours is not None:
                time_since_timeout_reset = current_time - timeout_start_time
                running_time = current_time - running_start_time
                if time_since_timeout_reset > timeout_hours * 3600:
                    logging.warning(f"Job {job_id} running time since last milestone "
                                  f"{time_since_timeout_reset/3600:.2f}h exceeds timeout {timeout_hours}h "
                                  f"(total running time: {running_time/3600:.2f}h)")
                    try:
                        subprocess.run(['scancel', job_id], check=False,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        logging.info(f"Job {job_id} cancelled due to timeout")
                    except Exception as e:
                        logging.error(f"Failed to cancel job {job_id}: {e}")
                    timed_out = True
                    break
            
            # MONITORING AGENT CHECK (only when job is running)
            if monitor and running_start_time is not None and job_state == "R":
                should_check = (
                    last_monitor_check is None or 
                    (current_time - last_monitor_check) >= monitor_interval
                )
                
                if should_check:
                    monitor_check_count += 1
                    try:
                        assessment = monitor.assess_calculation(
                            calc_dir=calc_dir,
                            start_time=running_start_time,
                            last_check_time=last_monitor_check,
                            check_count=monitor_check_count,
                            physical_context=physical_context,
                        )
                        last_monitor_check = current_time
                        monitor_assessment = assessment
                        
                        logging.info(f"[Monitor] Status: {assessment.get('status')}")
                        
                        if assessment.get("details"):
                            logging.info(f"[Monitor] Details: {assessment['details']}")
                        
                        # Check for timeout reset signal from the monitor stack
                        if assessment.get("reset_timeout"):
                            timeout_start_time = current_time
                            timeout_reset_count += 1
                            logging.info(f"[Monitor] TIMEOUT RESET: {assessment.get('milestone_reached', '')}")
                            logging.info(f"[Monitor] Timeout counter reset (reset #{timeout_reset_count}). "
                                       f"New timeout window: {timeout_hours}h from now")
                        
                        if assessment.get("should_kill") or assessment.get("decision") == "terminate":
                            logging.warning(f"[Monitor] KILL signal: {assessment.get('failure_reason')}")
                            logging.warning(f"[Monitor] Suggested fix: {assessment.get('suggested_fix')}")
                            logging.warning(f"[Monitor] Details: {assessment.get('details')}")
                            
                            try:
                                subprocess.run(['scancel', job_id], check=False,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                logging.info(f"Job {job_id} cancelled by RuleMonitor")
                            except Exception as e:
                                logging.error(f"Failed to cancel job {job_id}: {e}")
                            
                            terminated_by_rule_monitor = True
                            break
                        
                        elif assessment.get("decision") == "converged":
                            logging.info(f"[Monitor] Calculation converged! Waiting for job to finish...")
                    
                    except Exception as e:
                        logging.warning(f"[Monitor] Assessment failed: {e}")
            
            # Log state with timing info
            if running_start_time is not None:
                running_time = current_time - running_start_time
                if timeout_reset_count > 0:
                    logging.info(f"Job {job_id} state: {job_state} (running for {running_time/60:.1f} min, "
                               f"timeout resets: {timeout_reset_count})")
                else:
                    logging.info(f"Job {job_id} state: {job_state} (running for {running_time/60:.1f} min)")
            else:
                pending_so_far = current_time - submit_time
                logging.info(f"Job {job_id} state: {job_state} (pending for {pending_so_far/60:.1f} min)")
            
            # Check for terminal states
            if job_state in ["CD", "CG", "F", "TO", "NF"]:
                logging.warning(f"Job {job_id} finished with non-running state: {job_state}")
                break

        except subprocess.CalledProcessError:
            logging.info(f"squeue check for {job_id} failed, assuming job is complete.")
            break
        except Exception as e:
            logging.error(f"Error checking job status for {job_id}: {e}")
            raise
        
        time.sleep(poll_interval)
    
    # Calculate final times
    end_time = time.time()
    if running_start_time is not None:
        running_time = end_time - running_start_time
    else:
        pending_time = end_time - submit_time
    
    total_time = pending_time + running_time
    
    logging.info(f"Job {job_id} timing - Pending: {pending_time/60:.1f} min, Running: {running_time/60:.1f} min")
    
    return {
        'pending_time': pending_time,
        'running_time': running_time,
        'total_time': total_time,
        'timed_out': timed_out,
        'terminated_by_rule_monitor': terminated_by_rule_monitor,
        'monitor_assessment': monitor_assessment
    }


# Alias for backward compatibility
wait_for_job = wait_for_slurm_job


# ============================================================
# LOCAL EXECUTION MODE
# ============================================================

def generate_local_script(job_name: str, cwd: str, vasp_version: str = None) -> str:
    """
    Generate a local execution script for VASP job.
    
    Args:
        job_name: Name for the job (used for logging)
        cwd: Directory where script will be created
        vasp_version: VASP version to use ("std", "ncl", "gam", or None for auto-detect)
        
    Returns:
        Path to the generated script
    """
    # Always use absolute path so the cd in the script works from any CWD
    cwd = os.path.abspath(cwd)
    script_path = os.path.join(cwd, "run_vasp_local.sh")

    # Auto-detect VASP version from INCAR if not specified
    if vasp_version is None:
        incar_path = os.path.join(cwd, "INCAR")
        vasp_version = detect_vasp_version(incar_path)
        logging.info(f"Auto-detected VASP version: {vasp_version}")
    
    # Get the appropriate VASP executable
    vasp_executable = get_vasp_command(vasp_version)
    
    # Check if mpirun is available
    has_mpirun = shutil.which('mpirun') is not None
    
    # VASP GPU version requires mpirun even for single process execution
    # to properly initialize MPI and OpenACC
    if has_mpirun:
        n_processes = max(1, LOCAL_MPI_PROCESSES)
        vasp_command = f"mpirun -np {n_processes} {vasp_executable}"
    else:
        # Fallback to direct execution (may not work with GPU version)
        vasp_command = vasp_executable
        logging.warning("mpirun not found - VASP GPU version may not work properly")
    
    # Generate environment setup section
    gpu_export = f"export CUDA_VISIBLE_DEVICES={GPU_ID}" if GPU_ID >= 0 else "# CUDA_VISIBLE_DEVICES: all GPUs visible (GPU_ID=-1)"
    if VASP_ENV_SCRIPT and os.path.exists(VASP_ENV_SCRIPT):
        env_setup = f"""# GPU device selection (must come before vasp_env.sh to override its default)
{gpu_export}
# Source VASP environment setup script
source "{VASP_ENV_SCRIPT}"
"""
    else:
        env_setup = f"""# GPU device selection
{gpu_export}
# No VASP environment script configured, using system defaults
"""
    
    script_content = f"""#!/bin/bash
# Local VASP execution script for: {job_name}
# Generated automatically - do not edit

{env_setup}
# Set OpenMP environment variables
export OMP_NUM_THREADS={LOCAL_OMP_THREADS}
export OMP_STACKSIZE=512m

# Run VASP calculation
echo "Starting VASP calculation at $(date)"
echo "Working directory: {cwd}"
echo "VASP command: {vasp_command}"

cd "{cwd}"
{vasp_command} > print-out 2>&1
EXIT_CODE=$?

echo "VASP calculation finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path


def submit_local_job(script_path: str, cwd: str) -> subprocess.Popen:
    """
    Start a local VASP job as a background process.
    
    Args:
        script_path: Path to the local execution script
        cwd: Working directory
        
    Returns:
        Popen object representing the running process
    """
    # Resolve to absolute paths so the script is findable after Popen changes CWD
    script_path = os.path.abspath(script_path)
    cwd = os.path.abspath(cwd)
    logging.info(f"Starting local job: {script_path}")

    # Open output files for stdout and stderr
    stdout_path = os.path.join(cwd, "vasp_run.out")
    stderr_path = os.path.join(cwd, "vasp_run.err")

    stdout_file = open(stdout_path, 'w')
    stderr_file = open(stderr_path, 'w')

    # Start the process
    process = subprocess.Popen(
        ['/bin/bash', script_path],
        cwd=cwd,
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True  # Detach from parent process group
    )
    
    logging.info(f"Local job started with PID: {process.pid} for {os.path.basename(cwd)}")
    
    # Store file handles on the process object for cleanup
    process._stdout_file = stdout_file
    process._stderr_file = stderr_file
    
    return process


def wait_for_local_job(
    process: subprocess.Popen,
    poll_interval: int = 10,
    timeout_hours: float = None,
    calc_dir: str = None,
    enable_monitoring: bool = True,
    monitor_interval: int = MONITOR_CHECK_INTERVAL,
    monitor_config: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
    physical_context: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Wait for a local VASP process to complete.
    
    Args:
        process: Popen object from submit_local_job
        poll_interval: Time in seconds between status checks (default: 10)
        timeout_hours: Maximum running time in hours. None means no timeout.
        calc_dir: Directory containing the calculation (required for monitoring)
        enable_monitoring: Whether to enable rule-monitor checks (default: True)
        monitor_interval: Interval between rule-based monitoring checks in seconds (default: 600)
    
    Returns:
        dict with keys:
            - pending_time: Always 0 for local jobs
            - running_time: Time spent running (seconds)  
            - total_time: Total time in seconds
            - timed_out: Whether the job was cancelled due to timeout
            - terminated_by_rule_monitor: Whether the rule monitor requested termination
            - monitor_assessment: Final assessment from the monitor stack (if enabled)
            - exit_code: Process exit code
    """
    pid = process.pid
    logging.info(f"Waiting for local job (PID: {pid}) to complete...")
    if timeout_hours:
        logging.info(f"Timeout limit: {timeout_hours} hours")
    
    # Initialize monitoring agent if enabled and calc_dir is provided
    monitor = None
    if enable_monitoring and calc_dir:
        try:
            from agents.reasoning_monitor import ReasoningMonitor

            # A fresh monitor per job (rule-monitor milestone state is per job), sharing
            # the workflow's LLM client so monitor calls count toward the token budget.
            monitor = ReasoningMonitor(config=monitor_config, llm_client=llm_client)
            logging.info(f"ReasoningMonitor enabled: checking OSZICAR/OUTCAR every {monitor_interval}s")
        except ImportError as e:
            logging.warning(f"Could not import ReasoningMonitor: {e}. Monitoring disabled.")
            monitor = None
    
    start_time = time.time()
    timeout_start_time = start_time  # Separate timer for timeout (can be reset)
    running_time = 0
    timed_out = False
    terminated_by_rule_monitor = False
    last_monitor_check = None
    monitor_assessment = None
    timeout_reset_count = 0  # Track how many times timeout was reset
    monitor_check_count = 0  # Number of rule-based monitor checks so far
    
    try:
        while True:
            # Check if process is still running
            poll_result = process.poll()
            if poll_result is not None:
                # Process has finished
                logging.info(f"Local job (PID: {pid}) completed with exit code: {poll_result}")
                break
            
            current_time = time.time()
            running_time = current_time - start_time
            time_since_timeout_reset = current_time - timeout_start_time
            
            # Check timeout (based on time since last reset, not total running time)
            if timeout_hours is not None and time_since_timeout_reset > timeout_hours * 3600:
                logging.warning(f"Local job (PID: {pid}) running time since last milestone "
                              f"{time_since_timeout_reset/3600:.2f}h exceeds timeout {timeout_hours}h "
                              f"(total running time: {running_time/3600:.2f}h)")
                try:
                    # Try graceful termination first
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    time.sleep(5)
                    if process.poll() is None:
                        # Force kill if still running
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    logging.info(f"Local job (PID: {pid}) terminated due to timeout")
                except (ProcessLookupError, PermissionError) as e:
                    logging.warning(f"Could not terminate process {pid}: {e}")
                timed_out = True
                break
            
            # MONITORING AGENT CHECK
            if monitor and calc_dir:
                should_check = (
                    last_monitor_check is None or 
                    (current_time - last_monitor_check) >= monitor_interval
                )
                
                if should_check:
                    monitor_check_count += 1
                    try:
                        assessment = monitor.assess_calculation(
                            calc_dir=calc_dir,
                            start_time=start_time,
                            last_check_time=last_monitor_check,
                            check_count=monitor_check_count,
                            physical_context=physical_context,
                        )
                        last_monitor_check = current_time
                        monitor_assessment = assessment
                        
                        logging.info(f"[Monitor] Status: {assessment.get('status')}")
                        
                        if assessment.get("details"):
                            logging.info(f"[Monitor] Details: {assessment['details']}")
                        
                        # Check for timeout reset signal from the monitor stack
                        if assessment.get("reset_timeout"):
                            timeout_start_time = current_time
                            timeout_reset_count += 1
                            logging.info(f"[Monitor] TIMEOUT RESET: {assessment.get('milestone_reached', '')}")
                            logging.info(f"[Monitor] Timeout counter reset (reset #{timeout_reset_count}). "
                                       f"New timeout window: {timeout_hours}h from now")
                        
                        if assessment.get("should_kill") or assessment.get("decision") == "terminate":
                            logging.warning(f"[Monitor] KILL signal: {assessment.get('failure_reason')}")
                            logging.warning(f"[Monitor] Suggested fix: {assessment.get('suggested_fix')}")
                            logging.warning(f"[Monitor] Details: {assessment.get('details')}")
                            
                            try:
                                os.killpg(os.getpgid(pid), signal.SIGTERM)
                                time.sleep(5)
                                if process.poll() is None:
                                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                                logging.info(f"Local job (PID: {pid}) terminated by RuleMonitor")
                            except (ProcessLookupError, PermissionError) as e:
                                logging.warning(f"Could not terminate process {pid}: {e}")
                            
                            terminated_by_rule_monitor = True
                            break
                        
                        elif assessment.get("decision") == "converged":
                            logging.info(f"[Monitor] Calculation converged! Waiting for job to finish...")
                    
                    except Exception as e:
                        logging.warning(f"[Monitor] Assessment failed: {e}")
            
            # Log progress (include timeout reset count if any)
            if timeout_reset_count > 0:
                logging.info(f"Local job (PID: {pid}) running for {running_time/60:.1f} min "
                           f"(timeout resets: {timeout_reset_count})")
            else:
                logging.info(f"Local job (PID: {pid}) running for {running_time/60:.1f} min")
            
            time.sleep(poll_interval)
    
    finally:
        # Ensure the VASP process is terminated if still running
        if process.poll() is None:
            logging.warning(f"Cleaning up still-running local job (PID: {pid})")
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                time.sleep(5)
                if process.poll() is None:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                logging.info(f"Local job (PID: {pid}) terminated during cleanup")
            except (ProcessLookupError, PermissionError) as e:
                logging.warning(f"Could not terminate process {pid} during cleanup: {e}")
        # Clean up file handles
        if hasattr(process, '_stdout_file'):
            process._stdout_file.close()
        if hasattr(process, '_stderr_file'):
            process._stderr_file.close()
    
    # Calculate final times
    end_time = time.time()
    running_time = end_time - start_time
    
    logging.info(f"Local job (PID: {pid}) timing - Running: {running_time/60:.1f} min")
    
    return {
        'pending_time': 0,  # No pending time for local jobs
        'running_time': running_time,
        'total_time': running_time,
        'timed_out': timed_out,
        'terminated_by_rule_monitor': terminated_by_rule_monitor,
        'monitor_assessment': monitor_assessment,
        'exit_code': process.returncode
    }


# ============================================================
# EXCEPTIONS AND ENTRY POINT
# ============================================================

class JobTimeoutError(Exception):
    """Exception raised when a VASP job exceeds the timeout limit."""
    def __init__(self, message: str, timing_info: dict):
        super().__init__(message)
        self.timing_info = timing_info


class JobTerminatedByRuleMonitorError(Exception):
    """Exception raised when the rule monitor requests job termination."""
    def __init__(self, message: str, timing_info: dict, assessment: dict = None):
        super().__init__(message)
        self.timing_info = timing_info
        self.assessment = assessment


def run_vasp_job(
    job_name: str, 
    cwd: str, 
    timeout_hours: float = None,
    enable_monitoring: bool = True,
    monitor_interval: int = MONITOR_CHECK_INTERVAL,
    execution_mode: str = None,
    vasp_version: str = None,
    monitor_config: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
    physical_context: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Run a VASP job: generate script, submit job, and wait for completion.
    
    Automatically selects between Slurm and local execution based on
    configuration and system availability.
    
    This is a blocking call that handles the complete job lifecycle.
    
    Args:
        job_name: Name for the job
        cwd: Working directory containing VASP inputs
        timeout_hours: Maximum RUNNING time in hours (excluding pending time for Slurm). 
                      None means no timeout.
        enable_monitoring: Whether to enable rule-monitor checks (default: True)
        monitor_interval: Interval between rule-based monitoring checks in seconds (default: 600)
        execution_mode: Override execution mode ("slurm", "local", or None for auto-detect)
        vasp_version: VASP version to use ("std", "ncl", "gam", or None for auto-detect)
    
    Returns:
        dict with timing information:
            - pending_time: Time spent waiting (0 for local, may be >0 for Slurm)
            - running_time: Time spent actually running (seconds)
            - total_time: Total elapsed time (seconds)
            - timed_out: Whether the job was cancelled due to timeout
            - terminated_by_rule_monitor: Whether the rule monitor requested termination
            - monitor_assessment: Final assessment from the monitor stack
    
    Raises:
        JobTimeoutError: If the job exceeded the timeout limit
        JobTerminatedByRuleMonitorError: If the rule monitor detected a problem and terminated the job
    """
    # Determine execution mode
    mode = execution_mode if execution_mode else get_execution_mode()
    logging.info(f"Running VASP job '{job_name}' in {mode.upper()} mode")

    if mode == "slurm":
        # Slurm execution path
        script_path = generate_slurm_script(job_name, cwd, vasp_version=vasp_version)
        job_id = submit_slurm_job(script_path, cwd)
        result = wait_for_slurm_job(
            job_id, 
            timeout_hours=timeout_hours,
            calc_dir=cwd,
            enable_monitoring=enable_monitoring,
            monitor_interval=monitor_interval,
            monitor_config=monitor_config,
            llm_client=llm_client,
            physical_context=physical_context,
        )
    else:
        # Local execution path
        script_path = generate_local_script(job_name, cwd, vasp_version=vasp_version)
        process = submit_local_job(script_path, cwd)
        result = wait_for_local_job(
            process, 
            timeout_hours=timeout_hours,
            calc_dir=cwd,
            enable_monitoring=enable_monitoring,
            monitor_interval=monitor_interval,
            monitor_config=monitor_config,
            llm_client=llm_client,
            physical_context=physical_context,
        )
    
    # Handle timeout
    if result['timed_out']:
        raise JobTimeoutError(
            f"VASP job {job_name} timed out after {result['running_time']/3600:.2f} hours of running "
            f"(pending time was {result['pending_time']/60:.1f} min, not counted)",
            timing_info=result
        )
    
    # Handle monitor kill
    if result.get('terminated_by_rule_monitor', False):
        assessment = result.get('monitor_assessment', {})
        failure_reason = assessment.get('failure_reason', 'UNKNOWN')
        suggested_fix = assessment.get('suggested_fix', 'NONE')
        raise JobTerminatedByRuleMonitorError(
            f"VASP job {job_name} terminated by RuleMonitor: {failure_reason}. "
            f"Suggested fix: {suggested_fix}. "
            f"Running time was {result['running_time']/60:.1f} min.",
            timing_info=result,
            assessment=assessment
        )
    
    logging.info(f"VASP job {job_name} completed in {cwd}")
    logging.info(f"  Execution mode: {mode.upper()}")
    logging.info(f"  Pending time: {result['pending_time']/60:.1f} min")
    logging.info(f"  Running time: {result['running_time']/60:.1f} min")
    
    return result
