import random
from typing import Generator, Tuple
import numpy as np
import torch


def batch(array: list, batch_size: int) -> Generator[Tuple[int, list, int], None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(array), batch_size):
        yield i, array[i : i + batch_size], len(array[i : i + batch_size])


def random_derangement(lst):
    n = len(lst)
    while True:
        result = list(range(n))
        for i in range(n - 1):
            j = random.randrange(i + 1, n)
            result[i], result[j] = result[j], result[i]
        if all(i != result[i] for i in range(n)):
            return [lst[result[i]] for i in range(n)]


def first_match(lst, condition):
    for item in lst:
        if condition(item):
            return item
    return None


def get_elbow_index(data: np.array) -> int:
    """Find the elbow point in a 1D array using the second derivative method."""

    dy = np.diff(data)
    d2y = np.diff(dy)

    elbow_index = np.argmax(np.abs(d2y)) + 2

    return elbow_index


def get_logprobs_diff_elbow(
    logprobs_diffs: dict[tuple[str, str], torch.Tensor], langs: list[tuple[str, str]]
) -> int:
    mean_diffs_cumsum = (
        torch.stack([logprobs_diffs[lang].mean(dim=-1) for lang in langs])
        .mean(dim=0)
        .clamp(min=0)
        .flatten()
        .sort(descending=True)[0]
        .cumsum(dim=0)
    )

    return get_elbow_index(mean_diffs_cumsum.numpy()) + 1


def get_top_k(logprobs_diff: torch.Tensor, top_k: int) -> set[tuple[int, int]]:
    """
    Get the best heads contributing to the logprobs difference.

    Args:
        logprobs_diff (torch.Tensor): Tensor of shape (num_layers, num_heads)
        top_k (int): Number of top heads to return

    Returns:
        dict[tuple[int, int], float]: Dictionary mapping (layer, head) to their average contribution
    """

    heads = set()
    _, indices = logprobs_diff.flatten().topk(top_k)

    for idx in indices:
        layer = idx // logprobs_diff.shape[1]
        head = idx % logprobs_diff.shape[1]

        heads.add((layer.item(), head.item()))

    return sorted(heads, key=lambda x: x[0])
