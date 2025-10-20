import random
from typing import Generator, Tuple

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
