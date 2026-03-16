#!/usr/bin/env python3
"""Print statistics from stored runs."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.records import RunStore


def main():
    parser = argparse.ArgumentParser(description="Run statistics")
    parser.add_argument("--runs-dir", type=str, default="data/runs")
    args = parser.parse_args()

    store = RunStore(store_dir=args.runs_dir)
    index = store.load_index()

    if not index:
        print("No runs found.")
        return

    print(f"Total runs: {len(index)}")
    by_outcome = {}
    by_user = {}
    for r in index:
        outcome = r.get("outcome", "unknown")
        user = r.get("user_id", "unknown")
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        by_user[user] = by_user.get(user, 0) + 1

    print("\nBy outcome:")
    for k, v in sorted(by_outcome.items()):
        print(f"  {k}: {v}")

    print("\nBy user:")
    for k, v in sorted(by_user.items()):
        print(f"  {k}: {v}")

    # Step duration summary from successful runs
    success_runs = [r for r in index if r.get("outcome") == "success"]
    if success_runs:
        print(f"\nStep durations (from {len(success_runs)} successful runs):")
        all_durations: dict[str, list[int]] = {}
        for r in success_runs:
            for step, dur in r.get("step_durations", {}).items():
                all_durations.setdefault(step, []).append(dur)
        for step in sorted(all_durations.keys(), key=int):
            durs = all_durations[step]
            avg = sum(durs) / len(durs)
            print(f"  Step {step}: avg={avg:.0f} frames, "
                  f"min={min(durs)}, max={max(durs)}, n={len(durs)}")


if __name__ == "__main__":
    main()
