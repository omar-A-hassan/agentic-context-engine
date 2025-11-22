"""
ClaudeCodeEnvironment - Integration Pattern environment for Claude Code CLI.

This environment uses the ACE Integration Pattern where Claude Code is the agent:
1. INJECT: Playbook context added to Claude Code's prompt
2. EXECUTE: Claude Code runs with learned strategies as context
3. LEARN: ACE Reflector/Curator learn from Claude Code's execution

Claude Code acts as the generator - it receives playbook context and executes tasks.
No separate PassthroughGenerator needed.
"""

import subprocess
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ace import TaskEnvironment, Sample, EnvironmentResult, Playbook, GeneratorOutput
from ace.integrations.base import wrap_playbook_context
from execution_trace_parser import ExecutionTraceParser, parse_and_format_execution


class ClaudeCodeEnvironment(TaskEnvironment):
    """Environment that evaluates tasks by executing via Claude Code."""

    def __init__(self, workspace_dir: str):
        """
        Initialize environment.

        Args:
            workspace_dir: Directory where Claude Code will work
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.parser = ExecutionTraceParser()

    def _parse_strategies_from_commit(self, commit_message: str) -> List[str]:
        """
        Extract strategy IDs from commit message.

        Looks for patterns like:
        - "Applied strategies: strategy-00001, strategy-00005"
        - "Applied strategies: none"

        Args:
            commit_message: Git commit message text

        Returns:
            List of strategy IDs found in commit
        """
        # Look for "Applied strategies:" pattern
        pattern = r'Applied strategies:\s*([^\n]+)'
        match = re.search(pattern, commit_message, re.IGNORECASE)

        if not match:
            return []

        strategies_text = match.group(1).strip()

        # Handle "none" case
        if strategies_text.lower() in ['none', 'n/a', 'not applicable']:
            return []

        # Extract strategy IDs (pattern: strategy-00001, testing_setup-00003, etc.)
        strategy_ids = re.findall(r'[\w_]+-\d+', strategies_text)
        return strategy_ids

    def _extract_reasoning_from_output(self, output: str) -> str:
        """
        Extract Claude Code's reasoning/explanation from its output.

        Claude Code typically explains what it's doing before taking action.
        This extracts the explanatory text.

        Args:
            output: Claude Code's stdout

        Returns:
            Extracted reasoning text
        """
        # Claude Code output often has explanations before tool use
        # Extract the first substantial paragraph as reasoning
        lines = output.strip().split('\n')
        reasoning_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                if reasoning_lines:  # Stop at first empty line after content
                    break
                continue

            # Collect reasoning until we hit tool output markers
            if any(marker in line for marker in ['```', '<function_calls>', 'Reading file', 'Writing file']):
                break

            reasoning_lines.append(line)

            # Stop after collecting a reasonable amount
            if len(reasoning_lines) >= 10:
                break

        return '\n'.join(reasoning_lines) if reasoning_lines else output[:500]

    def _extract_strategy_citations(self, stdout: str) -> str:
        """
        Extract strategy citations from Claude Code's output.

        Looks for "Applied Learned Strategies" or "Applied strategies" section.

        Args:
            stdout: Claude Code's full output

        Returns:
            Strategy citations string or default message
        """
        # Look for "Applied Strategies" or "Applied Learned Strategies" section
        pattern = r'###?\s*Applied (?:Learned )?Strategies(.*?)(?=###|$)'
        match = re.search(pattern, stdout, re.IGNORECASE | re.DOTALL)

        if match:
            strategies_section = match.group(1).strip()
            # Extract bullet points with strategy IDs
            strategy_lines = [line.strip() for line in strategies_section.split('\n')
                            if line.strip() and ('**[' in line or '- [' in line)]
            if strategy_lines:
                return '\n'.join(strategy_lines)

        return '(No strategies cited - outcome-based learning)'

    def _extract_final_summary(self, stdout: str) -> str:
        """
        Extract the final summary/conclusion from Claude Code output.

        Args:
            stdout: Claude Code's full output

        Returns:
            Summary text
        """
        # Look for "## Summary" section
        pattern = r'##\s*Summary(.*?)(?=##|$)'
        match = re.search(pattern, stdout, re.DOTALL)

        if match:
            summary = match.group(1).strip()
            # Take first paragraph or up to 300 chars
            if len(summary) > 300:
                return summary[:300] + "..."
            return summary

        # Fallback: take last substantial paragraph
        paragraphs = [p.strip() for p in stdout.split('\n\n') if len(p.strip()) > 50]
        if paragraphs:
            last_para = paragraphs[-1]
            if len(last_para) > 300:
                return last_para[:300] + "..."
            return last_para

        return "Task completed successfully"

    def _extract_environment_feedback(self, success: bool, git_diff: str) -> str:
        """
        Generate environment feedback message summarizing execution outcome.

        Args:
            success: Whether execution succeeded
            git_diff: Git diff of changes

        Returns:
            Environment feedback string
        """
        if not success:
            return "Task failed - see error output"

        # Count files changed
        files_changed = len(set(re.findall(r'diff --git a/(.*?) b/', git_diff)))

        # Count lines added/removed
        added = len(re.findall(r'^\+[^+]', git_diff, re.MULTILINE))
        removed = len(re.findall(r'^-[^-]', git_diff, re.MULTILINE))

        if files_changed > 0:
            return f"Translation task succeeded - {files_changed} files changed (+{added}/-{removed} lines)"
        else:
            return "Task completed successfully"

    def evaluate(self, sample: Sample, playbook: Playbook) -> Tuple[EnvironmentResult, GeneratorOutput]:
        """
        Execute task via Claude Code CLI using ACE Integration Pattern.

        Integration Pattern:
        1. INJECT: Format playbook context for Claude Code
        2. EXECUTE: Run Claude Code with learned strategies
        3. Return results for ACE to LEARN from

        Args:
            sample: The task sample
            playbook: Playbook with learned strategies

        Returns:
            Tuple of (EnvironmentResult with feedback, GeneratorOutput for Reflector)
        """
        # INJECT: Format playbook context (Integration Pattern step 1)
        strategy_context = wrap_playbook_context(playbook) if playbook.bullets() else ""

        # EXECUTE: Build prompt for Claude Code (Integration Pattern step 2)
        prompt = f"""Your job is to translate ACE framework from Python to TypeScript.

You have access to:
- Python ACE source code in source/
- TypeScript translation target in target/
- Translation specification in specs/project.md
- Coding standards in specs/rules.md

{strategy_context}

CRITICAL REQUIREMENTS:
1. Review the strategies listed above - they were learned from previous successful translations
2. Apply strategies that are relevant to this specific task
3. **Make a commit after completing each logical unit of work** (e.g., one file, one class)
4. Each commit should be atomic and buildable

WORKING DIRECTORY STRUCTURE:
- Use .agent/ as your scratchpad for planning and notes
- Keep track of your current status in .agent/TODO.md
- Store long-term plans in .agent/PLAN.md
- Create target/ for TypeScript output

TESTING HEURISTIC:
- Spend 80% of time on actual translation
- Spend 20% of time on tests
- Prioritize functional equivalence over perfect test coverage initially

TASK:
{sample.question}

CONTEXT:
{sample.context if sample.context else 'None'}

CRITICAL - SINGLE TASK EXECUTION:
Complete ONLY the task specified above, then STOP.
Do NOT continue to additional tasks from TODO.md.
The ACE learning loop will call you again for the next task.

Why: After each task, the Reflector analyzes what worked and the Curator updates
the playbook. The next task will have improved strategies based on your current work.

FOCUS ON TRANSLATION WORK:
Your PRIMARY goal is translating Python code to TypeScript, not project setup.

DO NOT spend excessive time on:
- Elaborate documentation (README, ARCHITECTURE, MIGRATION guides)
- Complex linting configurations (basic ESLint is sufficient)
- CI/CD infrastructure (GitHub Actions, workflows)
- Empty directory structures
- Comprehensive project planning documents

DO focus your effort on:
- Reading Python source files
- Translating classes, functions, and types to TypeScript
- Writing tests for translated code
- Fixing TypeScript compilation errors

If the task involves setup/infrastructure, complete it minimally and move to translation work.

RESPONSE FORMAT REQUIREMENT:
Start your response by explaining your approach and reasoning BEFORE taking any action.
Your response should follow this structure:

## Approach
[Explain your implementation plan - what you'll create, which patterns you'll use,
how you'll structure the code, and WHY you're making these choices]

## Strategy Application
For each relevant strategy from the learned strategies above, explain:
- WHY it's applicable to this specific task
- HOW you'll use it in your implementation (with concrete details)
- WHAT specific problem or challenge it addresses

## Implementation
[Then proceed with the actual work using tools]

EXECUTION TRACE FORMAT (REQUIRED):
You MUST output your work in this exact format for each action you take:

```
--- Step N: [Brief Description] ---
Reasoning: [Why you're doing this step]
Action: [Tool](parameters)
Result: [What happened]
Pattern: [What pattern this demonstrates]
```

Example execution trace:
```
--- Step 1: Analyze Python Source ---
Reasoning: Need to understand the Bullet dataclass structure before translation
Action: Read('workspace/source/ace/playbook.py', lines=45-90)
Result: Found dataclass with 8 fields (id, section, content, 3 counters, 2 timestamps) and 2 methods
Pattern: Source code analysis for translation planning

--- Step 2: Create TypeScript Interface ---
Reasoning: TypeScript needs type definitions; interface provides structure for the class
Action: Write('workspace/target/src/playbook.ts', content='export interface BulletData {...}')
Result: Created interface with 8 typed fields matching Python dataclass
Pattern: Interface-first design for TypeScript data structures

--- Step 3: Implement TypeScript Class ---
Reasoning: Need concrete class implementation with constructor and methods
Action: Write('workspace/target/src/playbook.ts', content='export class Bullet implements BulletData {...}')
Result: Added Bullet class with constructor and all 8 properties
Pattern: Class implementation following interface contract
```

This trace format is CRITICAL - it enables the learning system to extract concrete implementation patterns from your actual tool usage, not just summaries.

Instructions:
1. First, explain your approach and reasoning (see RESPONSE FORMAT above)
2. Read relevant source files to understand the Python implementation
3. Implement the TypeScript translation following your explained approach
4. Write/update tests if needed (aim for 20% of effort)
5. Commit your changes with strategy acknowledgment in the commit message
6. Update TODO.md with progress
7. Provide a summary of what was accomplished and any challenges encountered
8. STOP after completing THIS task - do not continue to next tasks
"""

        # Save prompt for debugging
        prompt_file = self.workspace_dir / "prompt.md"
        prompt_file.write_text(prompt)
        print(f"💾 Saved prompt to {prompt_file}")

        # Execute via Claude Code (pass via stdin)
        print(f"🔄 Executing via Claude Code in {self.workspace_dir}...")

        try:
            import time
            start_time = time.time()

            result = subprocess.run(
                ["claude", "--print", "--verbose", "--dangerously-skip-permissions"],
                input=prompt,
                text=True,
                cwd=str(self.workspace_dir),
                capture_output=True,
                timeout=600  # 10 minute timeout
            )

            execution_time = time.time() - start_time
            success = result.returncode == 0
            debug_output = result.stderr  # Capture debug output for tool trace parsing

            # Get git diff if this is a git repo
            git_diff = ""
            try:
                diff_result = subprocess.run(
                    ["git", "diff", "HEAD~1", "HEAD"],
                    cwd=str(self.workspace_dir),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if diff_result.returncode == 0:
                    git_diff = diff_result.stdout
            except:
                git_diff = "(No git diff available)"

            # Get commit message if available
            commit_msg = ""
            try:
                commit_result = subprocess.run(
                    ["git", "log", "-1", "--pretty=%B"],
                    cwd=str(self.workspace_dir),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if commit_result.returncode == 0:
                    commit_msg = commit_result.stdout.strip()
            except:
                commit_msg = "(No commit message available)"

            # Parse strategy application from commit
            strategies_applied = self._parse_strategies_from_commit(commit_msg)

            # Parse execution into structured traces
            execution_trace = parse_and_format_execution(
                stdout=result.stdout,
                git_diff=git_diff,
                commit_msg=commit_msg,
                success=success,
                debug_output=debug_output
            )

            # Debug: Save execution trace to file for inspection
            trace_file = self.workspace_dir / ".agent" / "last_execution_trace.md"
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(execution_trace)
            print(f"📝 Saved execution trace to {trace_file}")

            # Debug: Save raw Claude Code output for reasoning extraction
            stdout_file = self.workspace_dir / ".agent" / "last_claude_output.txt"
            stdout_file.write_text(result.stdout)
            print(f"📝 Saved raw output to {stdout_file}")

            # Extract implementation patterns from diff
            patterns = self.parser.extract_patterns_from_diff(git_diff)

            # Build structured feedback for Reflector (matching browseruse format)
            # Include full Claude Code output instead of parsed execution trace
            feedback = f"""### Performance Data
Question: {sample.question}
Model Reasoning: Task completed in {execution_time:.1f}s
Duration: {execution_time:.1f}s

=== CLAUDE CODE OUTPUT ===

{result.stdout}

=== END CLAUDE CODE OUTPUT ===

Model Prediction: {self._extract_final_summary(result.stdout)}
Ground Truth: (none)
Environment Feedback: {self._extract_environment_feedback(success, git_diff)}

### Playbook Context
Strategies Applied:
{self._extract_strategy_citations(result.stdout)}

### Git Diff (Reference)
{git_diff[:2000] if git_diff else '(No changes)'}
"""

            if not success:
                feedback += f"\n\n## ERROR OUTPUT\n{result.stderr}"

            # Create EnvironmentResult with feedback
            env_result = EnvironmentResult(
                feedback=feedback,
                ground_truth=sample.ground_truth,
                metrics={
                    "success": success,
                    "returncode": result.returncode,
                    "output_length": len(result.stdout),
                    "diff_size": len(git_diff),
                    "commit_message": commit_msg,
                    "strategies_applied": strategies_applied,
                    "strategies_applied_count": len(strategies_applied),
                }
            )

            # Create minimal GeneratorOutput for Reflector (Integration Pattern step 3)
            # External agents don't cite bullets beforehand - Reflector extracts them from output
            generator_output = GeneratorOutput(
                reasoning=f"Task: {sample.question}",
                final_answer=result.stdout,  # Claude Code's actual output
                bullet_ids=[],  # External agents don't pre-select bullets
                raw={"success": success, "execution_time": execution_time}
            )

            return env_result, generator_output

        except subprocess.TimeoutExpired:
            env_result = EnvironmentResult(
                feedback=f"TIMEOUT: Claude Code execution exceeded 10 minutes",
                ground_truth=sample.ground_truth,
                metrics={"success": False, "returncode": -1, "error": "timeout"}
            )
            generator_output = GeneratorOutput(
                reasoning=f"Task: {sample.question}",
                final_answer="Execution timed out",
                bullet_ids=[],
                raw={"success": False, "error": "timeout"}
            )
            return env_result, generator_output
        except Exception as e:
            env_result = EnvironmentResult(
                feedback=f"ERROR: {str(e)}",
                ground_truth=sample.ground_truth,
                metrics={"success": False, "returncode": -1, "error": str(e)}
            )
            generator_output = GeneratorOutput(
                reasoning=f"Task: {sample.question}",
                final_answer=f"Execution failed: {str(e)}",
                bullet_ids=[],
                raw={"success": False, "error": str(e)}
            )
            return env_result, generator_output
