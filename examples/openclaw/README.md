# OpenClaw + ACE Integration

Learn from [OpenClaw](https://docs.openclaw.ai) session transcripts and inject
learned strategies back into the agent via `AGENTS.md`.

## How it works

```
OpenClaw runs sessions  -->  JSONL transcripts on disk
                                     |
                        learn_from_traces.py (cron)
                                     |
                        TraceAnalyser (Reflect -> Update -> Apply)
                                     |
                     +---------+-----+--------+
                     |                        |
              ace_skillbook.json        AGENTS.md (updated)
                                              |
                              OpenClaw loads on next session
```

1. OpenClaw writes session transcripts to `~/.openclaw/agents/<id>/sessions/*.jsonl`
2. The script reads new sessions, parses them into trace dicts
3. `TraceAnalyser` runs the ACE learning pipeline (no agent execution needed)
4. The updated skillbook is saved and synced into `AGENTS.md` between marker comments

## Setup

```bash
# Install core dependencies
uv sync

# Set your LLM API key
export ANTHROPIC_API_KEY="your-key"
```

## Usage

```bash
# One-off run
uv run python examples/openclaw/learn_from_traces.py

# Dry run — parse sessions without learning
uv run python examples/openclaw/learn_from_traces.py --dry-run

# Reprocess all sessions (ignore processed log)
uv run python examples/openclaw/learn_from_traces.py --reprocess
```

### Cron

```bash
*/30 * * * * cd /path/to/agentic-context-engine && uv run python examples/openclaw/learn_from_traces.py >> /tmp/ace-openclaw.log 2>&1
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `ACE_MODEL` | `anthropic/claude-sonnet-4-20250514` | LLM for reflection and skill extraction |
| `OPENCLAW_AGENT_ID` | `main` | OpenClaw agent to learn from |
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw home directory |
| `OPENCLAW_WORKSPACE` | `~/.openclaw/workspace` | Workspace where AGENTS.md lives |

## Files produced

| Path | Description |
|---|---|
| `~/.openclaw/ace_skillbook.json` | Persistent skillbook (survives across runs) |
| `~/.openclaw/ace_processed.txt` | Log of already-processed session filenames |
| `~/.openclaw/workspace/AGENTS.md` | Updated with learned strategies between `<!-- ACE:SKILLBOOK -->` markers |

## Adapting the JSONL parser

The session JSONL format depends on your OpenClaw version. Inspect a real
session file to see the actual schema:

```bash
head -20 ~/.openclaw/agents/main/sessions/*.jsonl
```

Then adjust `parse_session_jsonl()` in the script to match.
