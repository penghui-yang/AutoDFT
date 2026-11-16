"""
Enhanced workflow data models for flexible DFT pipeline.

These models support dynamic workflow generation with detailed step definitions,
sanity checks, failure modes, and postprocessing rules.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class GlobalSettings:
    """
    Global computational settings that apply across the workflow.
    
    Attributes:
        ENCUT: Plane-wave cutoff energy (eV)
        PREC: Precision level (Accurate, Normal, etc.)
        EDIFF: Electronic convergence criterion
        EDIFFG: Ionic convergence criterion
        ISPIN: Spin polarization (1=non-spin, 2=spin-polarized)
        NBANDS_factor: Multiplier for number of bands
        ALGO: Electronic minimization algorithm
        LREAL: Real-space projection
        additional_settings: Any other global parameters
    """
    ENCUT: Optional[int] = None
    PREC: Optional[str] = None
    EDIFF: Optional[float] = None
    EDIFFG: Optional[float] = None
    ISPIN: Optional[int] = None
    NBANDS_factor: Optional[float] = None
    ALGO: Optional[str] = None
    LREAL: Optional[Union[str, bool]] = None
    additional_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """
    Detailed definition of a single workflow step.
    
    This is much richer than the old CalculationStep, containing all information
    needed to execute the step autonomously.
    
    Attributes:
        name: Step identifier (e.g., "relaxation", "scf_static", "bands_nscf")
        description: Human-readable description of the step's purpose
        input_files_needed: List of required input files
        output_files_produced: List of expected output files
        INCAR: INCAR parameters specific to this step
        KPOINTS: KPOINTS specification (string description or dict)
        POTCAR_notes: Notes about POTCAR requirements
        vaspkit_actions: List of vaspkit commands/actions to perform
        sanity_checks: List of validation checks to perform
        expected_time_estimate: Rough estimate of computation time
        depends_on: List of step names this step depends on
        agent_type: Which agent should execute this step (default: computing)
        additional_params: Any other step-specific parameters
        
    Extended attributes for advanced workflows:
        can_parallelize: Whether this step can run in parallel with siblings
        condition: Python expression for conditional execution
        branch_key: Key for grouping parallel branches
        is_parallel_set: For steps with multiple parallel sub-calculations (NEB)
        parallel_count: Number of parallel sub-calculations
        file_dependencies: Explicit file-to-step mapping
    """
    name: str
    description: str
    input_files_needed: List[str] = field(default_factory=list)
    output_files_produced: List[str] = field(default_factory=list)
    INCAR: Dict[str, Any] = field(default_factory=dict)
    KPOINTS: Union[str, Dict[str, Any]] = ""
    POTCAR_notes: str = ""
    vaspkit_actions: List[str] = field(default_factory=list)
    sanity_checks: List[str] = field(default_factory=list)
    expected_time_estimate: str = ""
    depends_on: List[str] = field(default_factory=list)
    agent_type: str = "computing"  # computing, postprocessing, validation, etc.
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    # Extended attributes for DAG-based workflows
    can_parallelize: bool = False  # Can run in parallel with sibling steps
    condition: Optional[str] = None  # Conditional execution expression
    branch_key: Optional[str] = None  # For grouping parallel branches
    is_parallel_set: bool = False  # E.g., NEB images
    parallel_count: int = 1  # Number of parallel sub-calculations
    file_dependencies: Dict[str, str] = field(default_factory=dict)  # file -> source_step


@dataclass
class PostprocessingRules:
    """
    Postprocessing and data extraction rules.
    
    Attributes:
        vaspkit_commands: List of vaspkit commands for postprocessing
        extraction_rules: Rules for extracting specific data from outputs
        plotting_commands: Commands for generating plots
        output_format: Desired output format (json, csv, etc.)
    """
    vaspkit_commands: List[str] = field(default_factory=list)
    extraction_rules: Dict[str, str] = field(default_factory=dict)
    plotting_commands: List[str] = field(default_factory=list)
    output_format: str = "json"


@dataclass
class FailureMode:
    """
    Definition of a potential failure mode and its recovery strategies.
    
    Attributes:
        issue: Description of the issue
        detection_criteria: How to detect this issue
        fixes: List of potential fixes to try
        priority: Priority order for attempting fixes
    """
    issue: str
    detection_criteria: str = ""
    fixes: List[str] = field(default_factory=list)
    priority: int = 1


@dataclass
class WorkflowPlan:
    """
    Complete workflow plan with all details needed for autonomous execution.
    
    This is the output from the enhanced PlanningAgent.
    
    Attributes:
        task_description: High-level task description
        material_id: Material identifier
        global_settings: Global computational settings
        workflow_steps: Ordered list of detailed workflow steps
        postprocessing: Postprocessing rules
        failure_modes: Known failure modes and recovery strategies
        metadata: Additional metadata
        estimated_total_time: Overall time estimate
        required_resources: Required computational resources
    """
    task_description: str
    material_id: str
    global_settings: GlobalSettings
    workflow_steps: List[WorkflowStep]
    postprocessing: PostprocessingRules = field(default_factory=PostprocessingRules)
    failure_modes: List[FailureMode] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_total_time: str = ""
    required_resources: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow plan to dictionary format."""
        return {
            "task_description": self.task_description,
            "material_id": self.material_id,
            "global_settings": self.global_settings.__dict__,
            "workflow_steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "input_files_needed": step.input_files_needed,
                    "output_files_produced": step.output_files_produced,
                    "INCAR": step.INCAR,
                    "KPOINTS": step.KPOINTS,
                    "POTCAR_notes": step.POTCAR_notes,
                    "vaspkit_actions": step.vaspkit_actions,
                    "sanity_checks": step.sanity_checks,
                    "expected_time_estimate": step.expected_time_estimate,
                    "depends_on": step.depends_on,
                    "agent_type": step.agent_type,
                    "additional_params": step.additional_params,
                }
                for step in self.workflow_steps
            ],
            "postprocessing": {
                "vaspkit_commands": self.postprocessing.vaspkit_commands,
                "extraction_rules": self.postprocessing.extraction_rules,
                "plotting_commands": self.postprocessing.plotting_commands,
                "output_format": self.postprocessing.output_format,
            },
            "failure_modes": [
                {
                    "issue": fm.issue,
                    "detection_criteria": fm.detection_criteria,
                    "fixes": fm.fixes,
                    "priority": fm.priority,
                }
                for fm in self.failure_modes
            ],
            "metadata": self.metadata,
            "estimated_total_time": self.estimated_total_time,
            "required_resources": self.required_resources,
        }

    def to_skeletal_plan(self) -> "SkeletalPlan":
        """Convert the legacy fully-parameterized plan to a skeletal plan."""
        steps = []
        for step in self.workflow_steps:
            steps.append(
                SkeletalStep(
                    step_name=step.name,
                    objective=step.description or step.name,
                    computation_type=step.additional_params.get("computation_type", step.agent_type),
                    depends_on=list(step.depends_on),
                    constraints=dict(step.additional_params.get("constraints", {})),
                    success_criteria={
                        "expected_outputs": list(step.output_files_produced),
                        "sanity_checks": list(step.sanity_checks),
                    },
                    priority=step.additional_params.get("priority", "required"),
                    condition=step.condition,
                    predefined_parameters=StepParameters(
                        incar=dict(step.INCAR),
                        kpoints=step.KPOINTS,
                        additional_files={},
                        vasp_version=step.additional_params.get("vasp_version", "std"),
                        reasoning_trace="Converted from legacy WorkflowPlan.",
                        confidence=1.0,
                        metadata={
                            "legacy_workflow_step": True,
                            "agent_type": step.agent_type,
                            "input_files_needed": list(step.input_files_needed),
                            "output_files_produced": list(step.output_files_produced),
                            "vaspkit_actions": list(step.vaspkit_actions),
                            "sanity_checks": list(step.sanity_checks),
                            "expected_time_estimate": step.expected_time_estimate,
                            "POTCAR_notes": step.POTCAR_notes,
                            "additional_params": dict(step.additional_params),
                            "file_dependencies": dict(step.file_dependencies),
                            "can_parallelize": step.can_parallelize,
                            "branch_key": step.branch_key,
                            "is_parallel_set": step.is_parallel_set,
                            "parallel_count": step.parallel_count,
                        },
                    ),
                )
            )

        return SkeletalPlan(
            task_description=self.task_description,
            material_formula=self.metadata.get("formula", self.material_id),
            material_type=self.metadata.get("material_type", "unknown"),
            overall_strategy=self.metadata.get("overall_strategy", "Converted from legacy workflow plan."),
            steps=steps,
            global_considerations=self.metadata.get("global_considerations", []),
            reasoning_trace=self.metadata.get("reasoning_trace", "Legacy workflow converted to skeletal plan."),
            material_id=self.material_id,
            metadata={
                "legacy_workflow_plan": True,
                "postprocessing": asdict(self.postprocessing),
                "failure_modes": [asdict(mode) for mode in self.failure_modes],
                "global_settings": asdict(self.global_settings) if self.global_settings else {},
                "estimated_total_time": self.estimated_total_time,
                "required_resources": self.required_resources,
                **self.metadata,
            },
        )


@dataclass
class StepParameters:
    """Single-step VASP parameters generated at runtime."""

    incar: Dict[str, Any] = field(default_factory=dict)
    kpoints: Union[str, Dict[str, Any]] = ""
    additional_files: Dict[str, str] = field(default_factory=dict)
    vasp_version: str = "std"
    reasoning_trace: str = ""
    confidence: float = 1.0
    fallback_parameters: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkeletalStep:
    """High-level workflow step without fixed execution parameters."""

    step_name: str
    objective: str
    computation_type: str
    depends_on: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    priority: str = "required"
    condition: Optional[str] = None
    predefined_parameters: Optional[StepParameters] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkeletalPlan:
    """Adaptive high-level plan used by the new controller."""

    task_description: str
    material_formula: str
    material_type: str
    overall_strategy: str
    steps: List[SkeletalStep] = field(default_factory=list)
    global_considerations: List[str] = field(default_factory=list)
    reasoning_trace: str = ""
    material_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOutcomeSummary:
    """Structured summary of an executed step for downstream reasoning."""

    step_name: str
    converged: bool
    total_energy: Optional[float] = None
    ionic_steps_count: int = 0
    electronic_steps_per_ionic: List[int] = field(default_factory=list)
    max_force: Optional[float] = None
    stress_tensor: Optional[List[List[float]]] = None
    symmetry_after: Optional[str] = None
    cell_volume_change_pct: Optional[float] = None
    bandgap_estimate: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    raw_oszicar_tail: str = ""
    raw_outcar_summary: str = ""
    timing: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningRecord:
    """Single reasoning event for planning, monitoring, recovery, or reflection."""

    stage: str
    input_context: Dict[str, Any]
    output_decision: Dict[str, Any]
    reasoning_trace: str = ""
    timestamp: str = ""
    model_name: str = ""
    token_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowReasoningLog:
    """Reasoning trace across the workflow lifecycle."""

    material_id: str
    records: List[ReasoningRecord] = field(default_factory=list)
    plan_revisions: List[Dict[str, Any]] = field(default_factory=list)

    def add_record(self, record: ReasoningRecord) -> None:
        self.records.append(record)


@dataclass
class ExecutionContext:
    """
    Extended execution context that tracks workflow progress.
    
    Attributes:
        material_id: Material identifier
        cif_path: Path to the original .cif file
        work_base_dir: Base directory for all calculations
        poscar_content: Current POSCAR content
        formula: Chemical formula
        completed_steps: Dictionary of completed step names and their output dirs (all steps)
        vasp_completed_steps: Dictionary of completed VASP computing step names and their output dirs (agent_type=="computing" with INCAR only)
        step_outputs: Dictionary of step outputs and extracted data
        metadata: Additional context data
        global_settings: Global settings from workflow plan
    """
    material_id: str
    cif_path: str
    work_base_dir: str
    poscar_content: Optional[str] = None
    formula: Optional[str] = None
    completed_steps: Dict[str, str] = field(default_factory=dict)  # step_name -> output_dir (all steps)
    vasp_completed_steps: Dict[str, str] = field(default_factory=dict)  # step_name -> output_dir (VASP computing steps only)
    step_outputs: Dict[str, Any] = field(default_factory=dict)  # step_name -> output data
    metadata: Dict[str, Any] = field(default_factory=dict)
    global_settings: Optional[GlobalSettings] = None
    step_outcomes: Dict[str, StepOutcomeSummary] = field(default_factory=dict)
    reasoning_log: Optional[WorkflowReasoningLog] = None
    current_skeletal_plan: Optional[SkeletalPlan] = None


@dataclass
class StepExecutionResult:
    """
    Result of executing a single workflow step.
    
    Attributes:
        step_name: Name of the executed step
        status: Execution status (success, failed, warning)
        output_dir: Directory containing outputs
        outputs_produced: List of files actually produced
        extracted_data: Data extracted from outputs
        logs: Execution logs
        error_message: Error description if failed
        warnings: List of warning messages
        metrics: Performance metrics (time, memory, etc.)
    """
    step_name: str
    status: str  # success, failed, warning
    output_dir: str
    outputs_produced: List[str] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    logs: str = ""
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringCheckpoint:
    """
    Checkpoint data for monitoring a running calculation.
    
    Attributes:
        calc_dir: Directory being monitored
        start_time: Unix timestamp when calculation started
        last_check_time: Unix timestamp of last monitoring check
        check_count: Number of monitoring checks performed
        assessment_history: List of assessment results
        current_status: Latest assessment status
    """
    calc_dir: str = ""
    start_time: float = 0.0
    last_check_time: Optional[float] = None
    check_count: int = 0
    assessment_history: List[Dict[str, Any]] = field(default_factory=list)
    current_status: str = "RUNNING"
    details: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    llm_assessment: Optional[str] = None
    
    def record_check(self, assessment_result: Dict[str, Any]):
        """Record a monitoring check result."""
        import time
        self.last_check_time = time.time()
        self.check_count += 1
        self.assessment_history.append(assessment_result)
        self.current_status = assessment_result.get("status", "UNKNOWN")


@dataclass
class RecoveryAttempt:
    """
    Record of a recovery/retry attempt.
    
    Attributes:
        attempt_number: Which retry this is (1-indexed)
        failure_reason: Why the previous attempt failed
        suggested_fix: The fix that was applied
        archive_path: Where failed files were archived
        modifications: INCAR modifications applied
        timestamp: When recovery was initiated
        success: Whether recovery setup succeeded
    """
    attempt_number: int
    failure_reason: str
    suggested_fix: str
    archive_path: Optional[str] = None
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: Optional[str] = None
    success: bool = False
    status: str = ""
    changes_applied: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    llm_reasoning: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class MonitoredStepResult(StepExecutionResult):
    """
    Extended step execution result with monitoring data.
    
    Attributes:
        monitoring_checkpoint: Monitoring state during execution
        recovery_attempts: List of recovery attempts made
        final_assessment: Final assessment from monitoring
        total_runtime: Total time including retries (seconds)
    """
    monitoring_checkpoint: Optional[MonitoringCheckpoint] = None
    recovery_attempts: List[RecoveryAttempt] = field(default_factory=list)
    final_assessment: Optional[Dict[str, Any]] = None
    total_runtime: float = 0.0
