"""Accuracy-only rule reward for both Counterfactual RLVR stages."""

from verl.utils.reward_score import geo3k


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    accuracy = float(geo3k.acc_reward(solution_str, ground_truth, use_boxed=True))
    return {"score": accuracy, "accuracy": accuracy}
