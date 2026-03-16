#!/usr/bin/env python3
"""Train/refit all learning models from stored run data.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --runs-dir data/runs --model-dir models/learned
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.records import RunStore
from src.learning.duration_model import StepDurationModel
from src.learning.transition_model import TransitionModel
from src.learning.anomaly_detector import AnomalyDetector
from src.learning.failure_classifier import FailureClassifier
from src.learning.online_updater import OnlineUpdater


def main():
    parser = argparse.ArgumentParser(description="Train learning models from run data")
    parser.add_argument("--runs-dir", type=str, default="data/runs")
    parser.add_argument("--model-dir", type=str, default="models/learned")
    args = parser.parse_args()

    store = RunStore(store_dir=args.runs_dir)
    index = store.load_index()

    total = len(index)
    success = sum(1 for r in index if r.get("outcome") == "success")
    failure = sum(1 for r in index if r.get("outcome") == "failure")
    abandoned = sum(1 for r in index if r.get("outcome") == "abandoned")

    print(f"Run store: {args.runs_dir}")
    print(f"  Total runs: {total}")
    print(f"  Success: {success}")
    print(f"  Failure: {failure}")
    print(f"  Abandoned: {abandoned}")
    print()

    if total == 0:
        print("No runs to train from. Run the pipeline and complete some tasks first.")
        return

    duration_model = StepDurationModel()
    transition_model = TransitionModel(num_steps=10)
    anomaly_detector = AnomalyDetector(method="knn")
    failure_classifier = FailureClassifier()

    updater = OnlineUpdater(
        duration_model=duration_model,
        transition_model=transition_model,
        anomaly_detector=anomaly_detector,
        failure_classifier=failure_classifier,
        run_store=store,
        model_dir=args.model_dir,
    )

    print("Fitting all models...")
    updater.initial_fit()

    # Print duration model stats
    print("\n--- Duration Model ---")
    stats = duration_model.serialize()
    for step, vals in sorted(stats.get("global", {}).items()):
        print(f"  Step {step}: mean={vals['mean']:.1f} std={vals['std']:.1f} "
              f"count={vals['count']} range=[{vals.get('min', '?'):.0f}, {vals.get('max', '?'):.0f}]")

    # Print transition matrix highlights
    print("\n--- Transition Model ---")
    import numpy as np
    matrix = transition_model._transition_matrix
    for i in range(min(10, matrix.shape[0])):
        row = matrix[i]
        top = np.argsort(row)[::-1][:3]
        transitions = ", ".join(f"{j}({row[j]:.2f})" for j in top if row[j] > 0.01)
        if transitions:
            print(f"  Step {i} -> {transitions}")

    # Print anomaly detector status
    print("\n--- Anomaly Detector ---")
    for step, buf in sorted(anomaly_detector._buffers.items()):
        print(f"  Step {step}: {len(buf)} reference states buffered")

    # Print failure classifier status
    print("\n--- Failure Classifier ---")
    if failure_classifier.is_trained:
        print("  Trained model available")
    else:
        print(f"  Not enough data (need 20+ runs with 5+ failures)")
        print(f"  Current: {total} runs, {failure} failures")

    print("\nModels saved to:", args.model_dir)
    updater._save_models()


if __name__ == "__main__":
    main()
