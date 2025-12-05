#!/usr/bin/env python3
"""
Analyze results from 4-run ACE learning experiment.

Displays learning curve showing how ACE improves across runs.
Reads metrics from .data/runs.json.

Usage:
    python analyze_runs.py
"""

import json
import os
from pathlib import Path

DEMO_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("ACE_DEMO_DATA_DIR", str(DEMO_DIR / ".data")))
RUNS_FILE = DATA_DIR / "runs.json"


def load_runs():
    """Load run data from .data/runs.json."""
    if not RUNS_FILE.exists():
        print(f"❌ No runs data found at {RUNS_FILE}")
        print("\nRun the experiment first:")
        print("  python run_experiment.py")
        return None

    with open(RUNS_FILE) as f:
        return json.load(f)


def print_runs_table(runs):
    """Print formatted table of all runs."""
    print("\n" + "=" * 70)
    print("📊 ACE LEARNING CURVE - ALL RUNS")
    print("=" * 70)
    print()

    # Header
    print(
        f"{'Run':<6} {'Tasks':<10} {'Failures':<12} {'Strategies':<12} {'Success':<10}"
    )
    print("-" * 70)

    # Rows
    for i, run in enumerate(runs, 1):
        tasks = run.get("task_count", 0)
        failures = run.get("total_failures", 0)
        strategies = run.get("skillbook_strategies", 0)
        success = "✅ Yes" if run.get("success", False) else "❌ No"

        print(f"{i:<6} {tasks:<10} {failures:<12} {strategies:<12} {success:<10}")

    print()


def print_improvement_summary(runs):
    """Show improvement from first to last run."""
    if len(runs) < 2:
        print("⚠️  Need at least 2 runs to show improvement")
        return

    print("=" * 70)
    print("📈 IMPROVEMENT ANALYSIS")
    print("=" * 70)
    print()

    first = runs[0]
    last = runs[-1]

    # Failures comparison
    first_failures = first.get("total_failures", 0)
    last_failures = last.get("total_failures", 0)
    failure_change = last_failures - first_failures

    print("Validation Failures:")
    print(f"  Run 1: {first_failures}")
    print(f"  Run {len(runs)}: {last_failures}")
    if failure_change < 0:
        print(f"  📉 Improved by {abs(failure_change)} failures")
    elif failure_change > 0:
        print(f"  📈 Increased by {failure_change} failures")
    else:
        print(f"  ➡️  No change")
    print()

    # Strategies comparison
    first_strategies = first.get("skillbook_strategies", 0)
    last_strategies = last.get("skillbook_strategies", 0)
    strategy_growth = last_strategies - first_strategies

    print("Skillbook Strategies:")
    print(f"  Run 1: {first_strategies}")
    print(f"  Run {len(runs)}: {last_strategies}")
    if strategy_growth > 0:
        print(f"  📚 Learned {strategy_growth} new strategies")
    else:
        print(f"  ➡️  No new strategies")
    print()

    # Success rate
    successful_runs = sum(1 for r in runs if r.get("success", False))
    success_rate = (successful_runs / len(runs)) * 100

    print("Success Rate:")
    print(f"  {successful_runs}/{len(runs)} runs succeeded ({success_rate:.0f}%)")
    print()


def print_detailed_runs(runs):
    """Show detailed info for each run."""
    print("=" * 70)
    print("🔍 DETAILED RUN INFORMATION")
    print("=" * 70)
    print()

    for i, run in enumerate(runs, 1):
        print(f"Run {i}:")
        print(f"  Timestamp: {run.get('timestamp', 'N/A')}")
        print(f"  Tasks: {run.get('task_count', 0)}")
        print(f"  Validation Failures: {run.get('total_failures', 0)}")
        print(f"  Skillbook Strategies: {run.get('skillbook_strategies', 0)}")
        print(f"  Success: {'✅ Yes' if run.get('success', False) else '❌ No'}")

        if "error" in run:
            print(f"  Error: {run['error']}")

        print()


def main():
    """Analyze and display results."""
    print("\n" + "=" * 70)
    print("🧪 ACE LEARNING EXPERIMENT - RESULTS ANALYSIS")
    print("=" * 70)
    print()

    runs = load_runs()
    if not runs:
        return

    print(f"📁 Data source: {RUNS_FILE}")
    print(f"📊 Total runs: {len(runs)}")

    # Show last 4 runs (the experiment)
    recent_runs = runs[-4:] if len(runs) >= 4 else runs

    print_runs_table(recent_runs)
    print_improvement_summary(recent_runs)
    print_detailed_runs(recent_runs)

    print("=" * 70)
    print("💡 KEY INSIGHT")
    print("=" * 70)
    print()
    print("ACE replaces manual prompt engineering with automatic learning.")
    print("Each run improves the skillbook, reducing failures and increasing success.")
    print()


if __name__ == "__main__":
    main()
