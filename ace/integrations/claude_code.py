"""
Claude Code integration for ACE framework.

This module provides ACEClaudeCode, a wrapper for Claude Code CLI
that automatically learns from execution feedback.

Example:
    from ace.integrations import ACEClaudeCode

    agent = ACEClaudeCode(working_dir="./my_project")
    result = agent.run(task="Refactor the auth module")
    agent.save_playbook("learned.json")

    # With async learning
    agent = ACEClaudeCode(working_dir="./project", async_learning=True)
    result = agent.run(task="Task 1")  # Returns immediately
    agent.wait_for_learning()  # Wait for learning to complete

    # With deduplication
    from ace import DeduplicationConfig
    agent = ACEClaudeCode(
        working_dir="./project",
        dedup_config=DeduplicationConfig(similarity_threshold=0.85)
    )
"""

import subprocess
import shutil
import json
import os
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
from dataclasses import dataclass

from ..llm_providers import LiteLLMClient
from ..playbook import Playbook
from ..roles import Reflector, Curator, GeneratorOutput
from ..prompts_v2_1 import PromptManager
from .base import wrap_playbook_context

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig, DeduplicationManager


# Check if claude CLI is available
CLAUDE_CODE_AVAILABLE = shutil.which("claude") is not None


@dataclass
class ClaudeCodeResult:
    """Result from Claude Code execution."""

    success: bool
    output: str
    execution_trace: str
    returncode: int
    error: Optional[str] = None


class ACEClaudeCode:
    """
    Claude Code with ACE learning capabilities.

    Executes tasks via Claude Code CLI and learns from execution.
    Drop-in wrapper that automatically:
    - Injects learned strategies into prompts
    - Reflects on execution results
    - Updates playbook with new learnings

    Usage:
        # Simple usage
        agent = ACEClaudeCode(working_dir="./project")
        result = agent.run(task="Add unit tests for utils.py")

        # Reuse across tasks (learns from each)
        agent = ACEClaudeCode(working_dir="./project")
        agent.run(task="Task 1")
        agent.run(task="Task 2")  # Uses Task 1 learnings
        agent.save_playbook("expert.json")

        # Start with existing knowledge
        agent = ACEClaudeCode(
            working_dir="./project",
            playbook_path="expert.json"
        )
        agent.run(task="New task")
    """

    def __init__(
        self,
        working_dir: str,
        ace_model: str = "gpt-4o-mini",
        ace_llm: Optional[LiteLLMClient] = None,
        ace_max_tokens: int = 2048,
        playbook: Optional[Playbook] = None,
        playbook_path: Optional[str] = None,
        is_learning: bool = True,
        timeout: int = 600,
        async_learning: bool = False,
        max_reflector_workers: int = 3,
        dedup_config: Optional["DeduplicationConfig"] = None,
    ):
        """
        Initialize ACEClaudeCode.

        Args:
            working_dir: Directory where Claude Code will execute
            ace_model: Model for ACE learning (Reflector/Curator)
            ace_llm: Custom LLM client for ACE (overrides ace_model)
            ace_max_tokens: Max tokens for ACE learning LLM
            playbook: Existing Playbook instance
            playbook_path: Path to load playbook from
            is_learning: Enable/disable ACE learning
            timeout: Execution timeout in seconds (default: 600)
            async_learning: Run learning in background (default: False)
            max_reflector_workers: Parallel Reflector threads (default: 3)
            dedup_config: Optional DeduplicationConfig for bullet deduplication
        """
        if not CLAUDE_CODE_AVAILABLE:
            raise RuntimeError(
                "Claude Code CLI not found. Install from: " "https://claude.ai/code"
            )

        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.is_learning = is_learning
        self.timeout = timeout
        self.async_learning = async_learning
        self.max_reflector_workers = max_reflector_workers
        self.dedup_config = dedup_config

        # Load or create playbook
        if playbook_path:
            self.playbook = Playbook.load_from_file(playbook_path)
        elif playbook:
            self.playbook = playbook
        else:
            self.playbook = Playbook()

        # Create ACE LLM (for Reflector/Curator)
        self.ace_llm = ace_llm or LiteLLMClient(
            model=ace_model, max_tokens=ace_max_tokens
        )

        # Create ACE learning components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.curator = Curator(
            self.ace_llm, prompt_template=prompt_mgr.get_curator_prompt()
        )

        # Initialize deduplication manager if config provided
        self._dedup_manager: Optional["DeduplicationManager"] = None
        if dedup_config:
            from ..deduplication import DeduplicationManager

            self._dedup_manager = DeduplicationManager(dedup_config)

        # Async learning state
        self._learning_queue: queue.Queue = queue.Queue()
        self._learning_thread: Optional[threading.Thread] = None
        self._stop_learning = threading.Event()
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._lock = threading.Lock()

        # Start async learning thread if enabled
        if async_learning:
            self._start_async_learning()

    def run(self, task: str, context: str = "") -> ClaudeCodeResult:
        """
        Execute task via Claude Code with ACE learning.

        Args:
            task: Task description for Claude Code
            context: Additional context (optional)

        Returns:
            ClaudeCodeResult with execution details
        """
        # 1. INJECT: Add playbook context if learning enabled and has bullets
        if self.is_learning and self.playbook.bullets():
            playbook_context = wrap_playbook_context(self.playbook)
            prompt = (
                f"{task}\n\n{context}\n\n{playbook_context}"
                if context
                else f"{task}\n\n{playbook_context}"
            )
        else:
            prompt = f"{task}\n\n{context}" if context else task

        # 2. EXECUTE: Run Claude Code
        result = self._execute_claude_code(prompt)

        # 3. LEARN: Run ACE learning if enabled
        if self.is_learning:
            if self.async_learning:
                # Queue learning task for background processing
                with self._lock:
                    self._tasks_submitted += 1
                self._learning_queue.put((task, result))
            else:
                # Synchronous learning
                self._learn_from_execution(task, result)

        return result

    def _execute_claude_code(self, prompt: str) -> ClaudeCodeResult:
        """Execute Claude Code CLI with prompt."""
        try:
            # Filter out ANTHROPIC_API_KEY so Claude Code uses subscription auth
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

            result = subprocess.run(
                [
                    "claude",
                    "--print",
                    "--output-format=stream-json",
                    "--verbose",
                    "--dangerously-skip-permissions",
                ],
                input=prompt,
                text=True,
                cwd=str(self.working_dir),
                capture_output=True,
                timeout=self.timeout,
                env=env,
            )

            execution_trace, summary = self._parse_stream_json(result.stdout)

            return ClaudeCodeResult(
                success=result.returncode == 0,
                output=summary,
                execution_trace=execution_trace,
                returncode=result.returncode,
                error=result.stderr[:500] if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ClaudeCodeResult(
                success=False,
                output="",
                execution_trace="",
                returncode=-1,
                error=f"Execution timed out after {self.timeout}s",
            )
        except Exception as e:
            return ClaudeCodeResult(
                success=False,
                output="",
                execution_trace="",
                returncode=-1,
                error=str(e),
            )

    def _parse_stream_json(self, stdout: str) -> Tuple[str, str]:
        """
        Parse stream-json output from Claude Code.

        Args:
            stdout: Raw stream-json output

        Returns:
            Tuple of (execution_trace, final_summary)
        """
        trace_parts = []
        final_text = ""
        step_num = 0

        for line in stdout.split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                if event_type == "assistant":
                    message = event.get("message", {})
                    for block in message.get("content", []):
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if text.strip():
                                    trace_parts.append(f"[Reasoning] {text[:300]}")
                                    final_text = text
                            elif block.get("type") == "tool_use":
                                step_num += 1
                                tool_name = block.get("name", "unknown")
                                tool_input = block.get("input", {})
                                # Format tool call
                                if tool_name in ["Read", "Glob", "Grep"]:
                                    target = tool_input.get(
                                        "file_path"
                                    ) or tool_input.get("pattern", "")
                                    trace_parts.append(
                                        f"[Step {step_num}] {tool_name}: {target}"
                                    )
                                elif tool_name in ["Write", "Edit"]:
                                    target = tool_input.get("file_path", "")
                                    trace_parts.append(
                                        f"[Step {step_num}] {tool_name}: {target}"
                                    )
                                elif tool_name == "Bash":
                                    cmd = tool_input.get("command", "")[:80]
                                    trace_parts.append(f"[Step {step_num}] Bash: {cmd}")
                                else:
                                    trace_parts.append(f"[Step {step_num}] {tool_name}")
            except json.JSONDecodeError:
                continue

        execution_trace = (
            "\n".join(trace_parts) if trace_parts else "(No trace captured)"
        )

        # Extract summary from final text
        if final_text:
            paragraphs = [p.strip() for p in final_text.split("\n\n") if p.strip()]
            summary = paragraphs[-1][:300] if paragraphs else final_text[:300]
        else:
            summary = f"Completed {step_num} steps"

        return execution_trace, summary

    def _learn_from_execution(self, task: str, result: ClaudeCodeResult):
        """Run ACE learning pipeline after execution."""
        # Create GeneratorOutput for Reflector
        generator_output = GeneratorOutput(
            reasoning=result.execution_trace,
            final_answer=result.output,
            bullet_ids=[],  # External agents don't pre-select bullets
            raw={
                "success": result.success,
                "returncode": result.returncode,
            },
        )

        # Build feedback
        status = "succeeded" if result.success else "failed"
        feedback = f"Claude Code task {status}"
        if result.error:
            feedback += f"\nError: {result.error}"

        # Run Reflector
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=None,
            feedback=feedback,
        )

        # Get similarity report for Curator if deduplication enabled
        similarity_report = None
        if self._dedup_manager:
            similarity_report = self._dedup_manager.get_similarity_report(self.playbook)

        # Run Curator (with similarity report if available)
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"task: {task}",
            progress=f"Claude Code: {task}",
            similarity_report=similarity_report,
        )

        # Update playbook with delta operations
        self.playbook.apply_delta(curator_output.delta)

        # Apply consolidation operations if deduplication enabled
        if self._dedup_manager and curator_output.raw:
            self._dedup_manager.apply_operations_from_response(
                curator_output.raw, self.playbook
            )

    def save_playbook(self, path: str):
        """Save learned playbook to file."""
        self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        """Load playbook from file."""
        self.playbook = Playbook.load_from_file(path)

    def get_strategies(self) -> str:
        """Get current playbook strategies as formatted text."""
        if not self.playbook.bullets():
            return ""
        return wrap_playbook_context(self.playbook)

    def enable_learning(self):
        """Enable ACE learning."""
        self.is_learning = True

    def disable_learning(self):
        """Disable ACE learning (execution only)."""
        self.is_learning = False

    def _start_async_learning(self):
        """Start the background learning thread."""
        if self._learning_thread is not None and self._learning_thread.is_alive():
            return

        self._stop_learning.clear()
        self._learning_thread = threading.Thread(
            target=self._learning_worker, daemon=True
        )
        self._learning_thread.start()

    def _learning_worker(self):
        """Background worker that processes learning tasks."""
        while not self._stop_learning.is_set():
            try:
                # Wait for a task with timeout to allow checking stop flag
                task, result = self._learning_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # Process learning
                self._learn_from_execution(task, result)
            finally:
                with self._lock:
                    self._tasks_completed += 1
                self._learning_queue.task_done()

    def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for async learning to complete.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if all learning completed, False if timeout reached

        Example:
            agent = ACEClaudeCode(working_dir="./project", async_learning=True)
            agent.run(task="Task 1")
            agent.run(task="Task 2")
            success = agent.wait_for_learning(timeout=60.0)
        """
        if not self.async_learning:
            return True

        try:
            # Use join with timeout if provided
            if timeout is not None:
                import time

                start = time.time()
                while not self._learning_queue.empty():
                    elapsed = time.time() - start
                    if elapsed >= timeout:
                        return False
                    time.sleep(0.1)
                return True
            else:
                self._learning_queue.join()
                return True
        except Exception:
            return False

    def stop_async_learning(self, wait: bool = True):
        """
        Stop async learning pipeline.

        Args:
            wait: If True, wait for current tasks to complete (default: True)

        Example:
            agent.stop_async_learning()
        """
        if not self.async_learning:
            return

        if wait:
            self.wait_for_learning()

        self._stop_learning.set()
        if self._learning_thread and self._learning_thread.is_alive():
            self._learning_thread.join(timeout=5.0)

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """
        Get async learning statistics.

        Returns:
            Dictionary with learning progress info:
            - async_learning: Whether async mode is enabled
            - tasks_submitted: Total tasks queued
            - tasks_completed: Tasks finished processing
            - pending: Tasks still being processed
            - queue_size: Tasks waiting in queue

        Example:
            stats = agent.learning_stats
            print(f"Pending: {stats['pending']}")
        """
        with self._lock:
            submitted = self._tasks_submitted
            completed = self._tasks_completed

        return {
            "async_learning": self.async_learning,
            "tasks_submitted": submitted,
            "tasks_completed": completed,
            "pending": submitted - completed,
            "queue_size": self._learning_queue.qsize(),
        }

    def __del__(self):
        """Cleanup async learning resources on deletion."""
        try:
            self.stop_async_learning(wait=False)
        except Exception:
            pass


__all__ = ["ACEClaudeCode", "ClaudeCodeResult", "CLAUDE_CODE_AVAILABLE"]
