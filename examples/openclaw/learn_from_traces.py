#!/usr/bin/env python3
"""Learn from OpenClaw session transcripts and sync strategies to AGENTS.md.

Reads OpenClaw session JSONL files, feeds them through the ACE learning
pipeline (TraceAnalyser), and writes the updated skillbook back into
OpenClaw's workspace so the agent picks up learned strategies on its
next session.

Designed to run manually or via cron.

Requirements:
    uv sync  # core deps only — no extras needed

Environment:
    # Required — LLM for reflection and skill extraction
    export ANTHROPIC_API_KEY="your-api-key"

    # Optional — override defaults
    export ACE_MODEL="anthropic/claude-sonnet-4-20250514"
    export OPENCLAW_AGENT_ID="main"
    export OPENCLAW_HOME="~/.openclaw"
    export OPENCLAW_WORKSPACE="~/.openclaw/workspace"

Usage:
    # One-off run
    uv run python examples/openclaw/learn_from_traces.py

    # Cron (every 30 minutes)
    */30 * * * * cd /path/to/agentic-context-engine && uv run python examples/openclaw/learn_from_traces.py
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Ensure project root is importable
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env")

from ace_next import (
    LiteLLMClient,
    Reflector,
    Skillbook,
    SkillManager,
    TraceAnalyser,
    wrap_skillbook_context,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENCLAW_HOME = Path(os.getenv("OPENCLAW_HOME", Path.home() / ".openclaw")).expanduser()
OPENCLAW_AGENT_ID = os.getenv("OPENCLAW_AGENT_ID", "main")
OPENCLAW_WORKSPACE = Path(
    os.getenv("OPENCLAW_WORKSPACE", OPENCLAW_HOME / "workspace")
).expanduser()

SESSIONS_DIR = OPENCLAW_HOME / "agents" / OPENCLAW_AGENT_ID / "sessions"
SKILLBOOK_PATH = OPENCLAW_HOME / "ace_skillbook.json"
PROCESSED_LOG = OPENCLAW_HOME / "ace_processed.txt"

MODEL = os.getenv("ACE_MODEL", "anthropic/claude-sonnet-4-20250514")

# AGENTS.md markers for the skillbook section
MARKER_START = "<!-- ACE:SKILLBOOK:START -->"
MARKER_END = "<!-- ACE:SKILLBOOK:END -->"


# ---------------------------------------------------------------------------
# Processed-session tracking
# ---------------------------------------------------------------------------


def load_processed() -> set[str]:
    """Load the set of already-processed session filenames."""
    if PROCESSED_LOG.exists():
        return set(PROCESSED_LOG.read_text().splitlines())
    return set()


def save_processed(processed: set[str]) -> None:
    """Persist the set of processed session filenames."""
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_LOG.write_text("\n".join(sorted(processed)) + "\n")


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def parse_session_jsonl(path: Path) -> dict | None:
    """Parse an OpenClaw session JSONL file into a trace dict.

    Returns a dict with keys expected by ReflectStep::

        {"question", "reasoning", "answer", "feedback", "ground_truth", "skill_ids"}

    Returns None if the session has no usable content.

    Note:
        The exact JSONL schema depends on your OpenClaw version.  Adjust
        field names below after inspecting a real session file::

            head -20 ~/.openclaw/agents/main/sessions/*.jsonl
    """
    events: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not events:
        return None

    # Collect messages by role
    user_messages: list[str] = []
    assistant_messages: list[str] = []
    tool_calls: list[dict] = []

    for event in events:
        role = event.get("role", "")
        content = event.get("content", "")

        if role == "user" and content:
            user_messages.append(str(content))
        elif role == "assistant" and content:
            assistant_messages.append(str(content))

        # Capture tool invocations for richer traces
        if event.get("type") == "tool":
            tool_calls.append(
                {
                    "name": event.get("name", "unknown"),
                    "input": str(event.get("input", ""))[:300],
                    "output": str(event.get("output", ""))[:300],
                }
            )

    if not user_messages:
        return None

    question = user_messages[0]
    answer = assistant_messages[-1] if assistant_messages else "(no response)"

    # Build reasoning from the full conversation
    reasoning_parts: list[str] = []
    for i, msg in enumerate(user_messages):
        reasoning_parts.append(f"[User] {msg[:500]}")
        if i < len(assistant_messages):
            reasoning_parts.append(f"[Assistant] {assistant_messages[i][:500]}")
    for tc in tool_calls:
        reasoning_parts.append(f"[Tool: {tc['name']}] {tc['input']} -> {tc['output']}")

    return {
        "question": question,
        "reasoning": "\n".join(reasoning_parts),
        "answer": answer,
        "skill_ids": [],
        "feedback": f"Session completed with {len(tool_calls)} tool calls",
        "ground_truth": None,
    }


# ---------------------------------------------------------------------------
# AGENTS.md sync
# ---------------------------------------------------------------------------


def sync_to_agents_md(skillbook: Skillbook) -> None:
    """Write the skillbook into AGENTS.md between marker comments.

    Creates AGENTS.md if it doesn't exist.  Replaces the section between
    ``<!-- ACE:SKILLBOOK:START -->`` and ``<!-- ACE:SKILLBOOK:END -->``
    markers, preserving all other content.
    """
    agents_md = OPENCLAW_WORKSPACE / "AGENTS.md"
    existing = agents_md.read_text() if agents_md.exists() else ""

    context = wrap_skillbook_context(skillbook)
    if not context:
        print("  Skillbook empty — skipping AGENTS.md sync")
        return

    ace_section = (
        f"{MARKER_START}\n"
        f"## Learned Strategies\n\n"
        f"These strategies were learned from your past sessions. Use relevant\n"
        f"ones to improve your responses. Cite strategy IDs (e.g.\n"
        f"[web-scraping-00001]) when you apply them.\n\n"
        f"{context}\n"
        f"{MARKER_END}"
    )

    if MARKER_START in existing:
        # Replace existing section
        pattern = re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END)
        content = re.sub(pattern, ace_section, existing, flags=re.DOTALL)
    else:
        content = existing.rstrip() + "\n\n" + ace_section + "\n"

    agents_md.parent.mkdir(parents=True, exist_ok=True)
    agents_md.write_text(content)
    print(f"  Synced {len(skillbook.skills())} strategies to {agents_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def section(name: str) -> None:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn from OpenClaw session transcripts and sync to AGENTS.md."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse sessions but skip learning and sync.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Ignore the processed log and reprocess all sessions.",
    )
    args = parser.parse_args()

    # -- Discover new sessions --
    section("Discovering sessions")
    if not SESSIONS_DIR.exists():
        print(f"  Sessions directory not found: {SESSIONS_DIR}")
        print("  Is OpenClaw installed and has the agent run at least once?")
        return

    processed = set() if args.reprocess else load_processed()
    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    new_sessions = [f for f in session_files if f.name not in processed]

    print(f"  Sessions dir:  {SESSIONS_DIR}")
    print(f"  Total sessions: {len(session_files)}")
    print(f"  Already processed: {len(processed)}")
    print(f"  New to process: {len(new_sessions)}")

    if not new_sessions:
        print("  Nothing new to learn from.")
        return

    # -- Parse into traces --
    section("Parsing sessions")
    traces: list[dict] = []
    skipped = 0
    for session_file in new_sessions:
        trace = parse_session_jsonl(session_file)
        if trace:
            traces.append(trace)
            print(f"  + {session_file.name}: {trace['question'][:60]}")
        else:
            skipped += 1

    print(f"  Parsed: {len(traces)}, Skipped (empty): {skipped}")

    if not traces:
        print("  No usable traces found.")
        return

    if args.dry_run:
        print("\n  --dry-run: stopping before learning.")
        return

    # -- Load or create skillbook --
    section("Loading skillbook")
    if SKILLBOOK_PATH.exists():
        skillbook = Skillbook.load_from_file(str(SKILLBOOK_PATH))
        print(
            f"  Loaded {len(skillbook.skills())} existing strategies from {SKILLBOOK_PATH}"
        )
    else:
        skillbook = Skillbook()
        print("  Starting with empty skillbook")

    skills_before = len(skillbook.skills())

    # -- Run learning --
    section(f"Learning from {len(traces)} traces")
    client = LiteLLMClient(model=MODEL)
    analyser = TraceAnalyser.from_roles(
        reflector=Reflector(client),
        skill_manager=SkillManager(client),
        skillbook=skillbook,
    )

    results = analyser.run(traces, epochs=1, wait=True)

    errors = [r for r in results if r.error]
    if errors:
        for e in errors:
            print(f"  ERROR: {e.failed_at}: {e.error}")
    print(f"  Processed: {len(results) - len(errors)}/{len(results)}")

    skills_after = len(skillbook.skills())
    new_skills = skills_after - skills_before
    print(f"  New strategies: {new_skills} (total: {skills_after})")

    if skills_after > 0:
        print("\n  Latest strategies:")
        for skill in skillbook.skills()[-3:]:
            print(f"    [{skill.id}] {skill.content[:70]}")

    # -- Save skillbook --
    section("Saving")
    SKILLBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    skillbook.save_to_file(str(SKILLBOOK_PATH))
    print(f"  Skillbook: {SKILLBOOK_PATH}")

    # -- Mark sessions processed --
    processed.update(f.name for f in new_sessions)
    save_processed(processed)
    print(f"  Processed log: {PROCESSED_LOG}")

    # -- Sync to AGENTS.md --
    section("Syncing to OpenClaw")
    sync_to_agents_md(skillbook)

    section("Done")


if __name__ == "__main__":
    main()
