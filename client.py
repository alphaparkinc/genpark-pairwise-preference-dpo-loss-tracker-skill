"""
Pairwise Preference DPO Loss Tracker Skill Client
Pure Python Standard Library implementation of Direct Preference Optimization (DPO) (Rafailov et al.).
Calculates implicit rewards r(x, y) = beta * log(pi(y|x) / ref(y|x)),
pairwise log-odds differences, DPO losses, and chosen-rejected margin tracking.
"""

import math
from typing import List, Dict, Any, Tuple


class DPOLossTracker:
    """
    Direct Preference Optimization loss and implicit reward tracker.
    L_DPO = - E_{(x, y_w, y_l)} [ log sigma( beta * log(pi(y_w|x)/ref(y_w|x)) - beta * log(pi(y_l|x)/ref(y_l|x)) ) ]
    """

    def __init__(self, beta: float = 0.1):
        """
        :param beta: Temperature / KL penalty parameter (typically 0.1 to 0.5).
        """
        self.beta = beta

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def compute_implicit_reward(self, policy_logprob: float, ref_logprob: float) -> float:
        """r(x, y) = beta * (log pi(y|x) - log ref(y|x))"""
        return self.beta * (policy_logprob - ref_logprob)

    def evaluate_pair(
        self,
        policy_chosen_logprob: float,
        ref_chosen_logprob: float,
        policy_rejected_logprob: float,
        ref_rejected_logprob: float
    ) -> Dict[str, float]:
        """
        Evaluate single pairwise preference sample (chosen y_w vs rejected y_l).
        """
        reward_w = self.compute_implicit_reward(policy_chosen_logprob, ref_chosen_logprob)
        reward_l = self.compute_implicit_reward(policy_rejected_logprob, ref_rejected_logprob)

        margin = reward_w - reward_l
        prob_chosen = self._sigmoid(margin)

        # DPO Loss = - log(prob_chosen)
        # Numerical protection: clip prob_chosen
        safe_p = max(1e-12, prob_chosen)
        loss = -math.log(safe_p)

        return {
            "reward_chosen": reward_w,
            "reward_rejected": reward_l,
            "reward_margin": margin,
            "preference_probability": prob_chosen,
            "dpo_loss": loss,
            "accuracy": 1.0 if margin > 0 else 0.0
        }

    def evaluate_batch(self, pairs: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Evaluate a batch of pairwise samples.
        Each sample dict must contain: pi_w, ref_w, pi_l, ref_l.
        """
        if not pairs:
            return {"loss": 0.0, "mean_margin": 0.0, "accuracy": 0.0, "sample_count": 0}

        total_loss = 0.0
        total_margin = 0.0
        correct_count = 0

        for p in pairs:
            res = self.evaluate_pair(p["pi_w"], p["ref_w"], p["pi_l"], p["ref_l"])
            total_loss += res["dpo_loss"]
            total_margin += res["reward_margin"]
            correct_count += int(res["accuracy"])

        n = len(pairs)
        return {
            "mean_loss": total_loss / n,
            "mean_margin": total_margin / n,
            "accuracy": correct_count / n,
            "sample_count": n
        }
