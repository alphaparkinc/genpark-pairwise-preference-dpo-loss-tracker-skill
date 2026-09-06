"""
Example usage of Pairwise Preference DPO Loss Tracker Skill.
"""

from client import DPOLossTracker


def main():
    print("=== Pairwise Preference DPO Loss Tracker Demonstration ===")
    tracker = DPOLossTracker(beta=0.1)

    # Simulated batch of preference pairs:
    # 1. Clear preference aligned with policy
    # 2. Borderline / subtle improvement
    # 3. Misaligned policy (rejected has higher logprob than chosen)
    batch = [
        {"pi_w": -2.1, "ref_w": -4.5, "pi_l": -5.2, "ref_l": -4.6},  # Strongly aligned
        {"pi_w": -3.0, "ref_w": -3.2, "pi_l": -3.5, "ref_l": -3.3},  # Subtle margin
        {"pi_w": -6.0, "ref_w": -3.0, "pi_l": -2.0, "ref_l": -3.0}   # Misaligned
    ]

    print(f"Evaluating {len(batch)} pairwise trajectory comparisons (beta={tracker.beta}):\n")
    for idx, p in enumerate(batch):
        res = tracker.evaluate_pair(p["pi_w"], p["ref_w"], p["pi_l"], p["ref_l"])
        print(f"Pair #{idx + 1}:")
        print(f"  Implicit Reward (Chosen):   {res['reward_chosen']:+.4f}")
        print(f"  Implicit Reward (Rejected): {res['reward_rejected']:+.4f}")
        print(f"  Reward Margin:              {res['reward_margin']:+.4f}")
        print(f"  DPO Loss:                   {res['dpo_loss']:.4f}")
        print(f"  Accuracy:                   {'PASS' if res['accuracy'] == 1.0 else 'FAIL'}\n")

    summary = tracker.evaluate_batch(batch)
    print("--- Batch Summary ---")
    print("Mean DPO Loss:   ", round(summary["mean_loss"], 4))
    print("Mean Margin:     ", round(summary["mean_margin"], 4))
    print("Pairwise Accuracy:", f"{summary['accuracy'] * 100:.1f}%")


if __name__ == "__main__":
    main()
