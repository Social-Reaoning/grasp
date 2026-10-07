"""Compute per-category GRASP-Bench accuracy from eval results.

Categories: Gaze (T1-T6), Gesture (G1-G6), Joint (J1-J4, stored as C1-C4).

Usage:
    python -m eval.metric_report eval/results/grasp_qwen3_vl.json
"""

import json
import sys

GROUPS = {
    "Gaze": ["T1", "T2", "T3", "T4", "T5", "T6"],
    "Gesture": ["G1", "G2", "G3", "G4", "G5", "G6"],
    "Joint": ["C1", "C2", "C3", "C4"],
}


def evaluate(path):
    with open(path) as f:
        records = json.load(f)

    by_cat = {}
    by_diff = {}
    for r in records:
        cat = r.get("category", "unknown")
        diff = r.get("difficulty", "unknown")
        correct = r.get("pred") == r.get("answer")

        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][1] += 1
        if correct:
            by_cat[cat][0] += 1

        by_diff.setdefault(diff, [0, 0])
        by_diff[diff][1] += 1
        if correct:
            by_diff[diff][0] += 1

    total_c, total_n = 0, 0
    for group_name, cats in GROUPS.items():
        group_c, group_n = 0, 0
        print(f"\n===== {group_name} =====")
        for cat in cats:
            if cat in by_cat:
                c, n = by_cat[cat]
                group_c += c
                group_n += n
                print(f"  {cat}: {c}/{n} = {c / n:.3f}")
        if group_n:
            print(f"  Avg: {group_c}/{group_n} = {group_c / group_n:.3f}")
        total_c += group_c
        total_n += group_n

    known = {c for cats in GROUPS.values() for c in cats}
    for cat in sorted(by_cat):
        if cat not in known:
            c, n = by_cat[cat]
            total_c += c
            total_n += n
            print(f"\n  {cat}: {c}/{n} = {c / n:.3f}")

    print("\n===== By Difficulty =====")
    for diff in ["easy", "medium", "hard"]:
        if diff in by_diff:
            c, n = by_diff[diff]
            print(f"  {diff}: {c}/{n} = {c / n:.3f}")

    if total_n:
        print(f"\n===== Overall: {total_c}/{total_n} = {total_c / total_n:.3f} =====")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m eval.metric_report <results.json>")
        sys.exit(1)
    evaluate(sys.argv[1])
