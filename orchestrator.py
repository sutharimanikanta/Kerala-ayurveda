# D:\Download1\kerala_ayurveda_content_pack_v1\orchestrator.py
# D:\Download1\kerala_ayurveda_content_pack_v1\orchestrator.py
"""
Main Orchestrator Agent
Autonomous multi-agent workflow controller with intent-aware retrieval support
"""

import logging
from typing import Dict
from datetime import datetime
from enum import Enum

from agents import OutlineAgent, WriterAgent, FactCheckerAgent, FinalizationAgent
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowMode(Enum):
    """Supported workflow modes"""
    QUESTION_ANSWERING = "question_answering"
    CONTENT_GENERATION = "content_generation"


class OrchestratorState(Enum):
    """Orchestrator execution states"""
    IDLE = "idle"
    PLANNING = "planning"
    WRITING = "writing"
    FACT_CHECKING = "fact_checking"
    REVISING = "revising"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class MainOrchestrator:
    """
    Main Orchestrator Agent

    Responsibilities:
    - Receive user intent (Q&A or content generation)
    - Automatically invoke agents in sequence
    - Handle fact-checker feedback loops
    - Enforce retry limits
    - Maintain execution trace for UI display
    - Support intent-aware retrieval modes
    """

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.max_retries = config.MAX_RETRIES

        # Initialize agents
        self.outline_agent = OutlineAgent(rag_pipeline)
        self.writer_agent = WriterAgent(rag_pipeline)
        self.fact_checker = FactCheckerAgent(rag_pipeline)
        self.finalizer = FinalizationAgent(rag_pipeline)

        # Execution state
        self.state = OrchestratorState.IDLE
        self.execution_trace = []
        self.retry_count = 0

        logger.info("✓ Orchestrator initialized")

    def execute_workflow(
        self,
        mode: str,
        user_input: Dict
    ) -> Dict:
        """
        Main workflow execution entry point

        Args:
            mode: "question_answering" or "content_generation"
            user_input: Dict with mode-specific parameters

        Returns:
            Final output with execution trace
        """
        self._reset_state()

        try:
            if mode == WorkflowMode.QUESTION_ANSWERING.value:
                return self._execute_qa_workflow(user_input)
            elif mode == WorkflowMode.CONTENT_GENERATION.value:
                return self._execute_content_workflow(user_input)
            else:
                raise ValueError(f"Unknown workflow mode: {mode}")

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            self.state = OrchestratorState.FAILED
            self._log_step("ERROR", f"Workflow failed: {str(e)}")
            raise

    def _execute_qa_workflow(self, user_input: Dict) -> Dict:
        """Execute Q&A workflow with intent-aware retrieval"""
        self.state = OrchestratorState.IDLE
        self._log_step("START", "Q&A workflow initiated")

        query = user_input.get('query', '')
        use_product_direct = user_input.get('use_product_direct', True)

        if not query:
            raise ValueError("Query is required for Q&A mode")

        # Intent-aware RAG retrieval and answer
        self._log_step(
            "RETRIEVING",
            f"Searching knowledge base for: {query} (mode: {'product-direct' if use_product_direct else 'balanced'})"
        )

        result = self.rag.answer_query(query, use_product_direct=use_product_direct)

        self.state = OrchestratorState.COMPLETED
        self._log_step("COMPLETED", "Q&A workflow finished")

        return {
            'mode': 'question_answering',
            'result': result,
            'execution_trace': self.execution_trace
        }

    def _execute_content_workflow(self, user_input: Dict) -> Dict:
        """Execute content generation workflow with automatic retries"""
        self._log_step("START", "Content generation workflow initiated")

        # Step 1: Generate Outline
        self.state = OrchestratorState.PLANNING
        self._log_step("PLANNING", "Outline Agent generating article structure...")

        outline = self.outline_agent.generate_outline(user_input)
        self._log_step("PLANNING_COMPLETE", f"Outline created with {len(outline['sections'])} sections")

        # Step 2: Generate Draft (with potential retries)
        draft = self._generate_draft_with_retries(outline)

        # Step 3: Finalize
        self.state = OrchestratorState.FINALIZING
        self._log_step("FINALIZING", "Finalization Agent preparing final output...")

        final_output = self.finalizer.finalize(draft['draft'], draft['fact_check'])
        self._log_step("FINALIZING_COMPLETE", "Final output ready")

        self.state = OrchestratorState.COMPLETED
        self._log_step("COMPLETED", "Content generation workflow finished successfully")

        return {
            'mode': 'content_generation',
            'outline': outline,
            'draft': draft['draft'],
            'fact_check': draft['fact_check'],
            'final_output': final_output,
            'execution_trace': self.execution_trace,
            'total_retries': self.retry_count
        }

    def _generate_draft_with_retries(self, outline: Dict) -> Dict:
        """
        Generate draft with automatic fact-check retry loop

        Returns:
            Dict with final draft and fact-check results
        """
        draft = None
        fact_check = None

        while self.retry_count <= self.max_retries:
            # Generate/Revise Draft
            if self.retry_count == 0:
                self.state = OrchestratorState.WRITING
                self._log_step("WRITING", "Writer Agent creating initial draft...")
                draft = self.writer_agent.generate_draft(outline)
                self._log_step("WRITING_COMPLETE", f"Draft created with {len(draft['paragraphs'])} paragraphs")
            else:
                self.state = OrchestratorState.REVISING
                self._log_step(
                    "REVISING",
                    f"Writer Agent revising draft (Attempt {self.retry_count + 1}/{self.max_retries + 1})..."
                )

                # Pass fact-check issues to writer for revision
                draft = self.writer_agent.revise_draft(
                    draft=draft,
                    fact_check_issues=fact_check['issues'],
                    outline=outline
                )
                self._log_step("REVISION_COMPLETE", "Draft revised based on fact-checker feedback")

            # Fact-Check Draft
            self.state = OrchestratorState.FACT_CHECKING
            self._log_step("FACT_CHECKING", "Fact-Checker Agent validating claims...")

            fact_check = self.fact_checker.check_draft(draft)

            high_severity_issues = [
                i for i in fact_check['issues']
                if i['severity'] == 'high'
            ]

            if not high_severity_issues:
                # Success - no high-severity issues
                self._log_step(
                    "FACT_CHECK_PASSED",
                    f"✓ All claims validated ({len(fact_check['issues'])} minor issues)"
                )
                break
            else:
                # Retry needed
                self._log_step(
                    "FACT_CHECK_FAILED",
                    f"⚠️ {len(high_severity_issues)} high-severity issues found"
                )

                for issue in high_severity_issues[:3]:  # Log first 3 issues
                    self._log_step(
                        "ISSUE",
                        f"  - {issue['type']}: {issue['message'][:100]}"
                    )

                self.retry_count += 1

                if self.retry_count > self.max_retries:
                    # Max retries reached
                    self.state = OrchestratorState.FAILED
                    self._log_step(
                        "ERROR",
                        f"Max retries ({self.max_retries}) reached. Unresolved issues remain."
                    )
                    raise Exception(
                        f"Failed to generate valid content after {self.max_retries} retries. "
                        f"{len(high_severity_issues)} high-severity issues remain."
                    )

        return {
            'draft': draft,
            'fact_check': fact_check
        }

    def _reset_state(self):
        """Reset orchestrator state for new workflow"""
        self.state = OrchestratorState.IDLE
        self.execution_trace = []
        self.retry_count = 0

    def _log_step(self, step_type: str, message: str):
        """Log execution step for UI display"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step_type': step_type,
            'message': message,
            'state': self.state.value
        }
        self.execution_trace.append(log_entry)
        logger.info(f"[{step_type}] {message}")

    def get_execution_summary(self) -> Dict:
        """Get summary of current execution"""
        return {
            'state': self.state.value,
            'total_steps': len(self.execution_trace),
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }