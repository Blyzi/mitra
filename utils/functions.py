from typing import Generator, Tuple


def batch(array: list, batch_size: int) -> Generator[Tuple[int, list, int], None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(array), batch_size):
        yield i, array[i : i + batch_size], len(array[i : i + batch_size])
