"""
Construction of the seven AutoDFT agents for one workflow controller.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.llm_client import LLMClient
from agents.postprocessing_agent import PostprocessingAgent
from agents.reasoning_monitor import ReasoningMonitor
from agents.reasoning_recovery import ReasoningRecovery
from agents.step_planner import StepPlanner
from agents.step_reflector import StepReflector
from agents.strategic_planner import StrategicPlanner
from agents.vasp_executor import VASPExecutor


class WorkflowOrchestrator:
    """Holds the agents of one workflow controller; all LLM agents share one client and token budget."""

    def __init__(self, work_base_dir: str, config: Optional[Dict[str, Any]] = None):
        self.work_base_dir = work_base_dir
        self.config = config or {}
        self.llm_client = LLMClient(self.config)

        self.strategic_planner = StrategicPlanner(work_base_dir, config=self.config, llm_client=self.llm_client)
        self.step_planner = StepPlanner(config=self.config, llm_client=self.llm_client)
        self.vasp_executor = VASPExecutor(work_base_dir, config=self.config, llm_client=self.llm_client)
        # The runtime monitor is instantiated per VASP job inside execution.job_manager
        # (it keeps per-job state); this instance documents the shared configuration.
        self.reasoning_monitor = ReasoningMonitor(config=self.config, llm_client=self.llm_client)
        self.reasoning_recovery = ReasoningRecovery(config=self.config, llm_client=self.llm_client)
        self.step_reflector = StepReflector(config=self.config, llm_client=self.llm_client)
        self.postprocessing_agent = PostprocessingAgent()

    def reset_workflow_budget(self) -> None:
        self.llm_client.reset_budget()

    def token_budget_exhausted(self) -> bool:
        return self.llm_client.token_budget.exhausted
