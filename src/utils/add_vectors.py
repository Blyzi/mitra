import torch


def add_vectors(
    v1: dict[tuple[int, int], torch.Tensor],
    v2: dict[tuple[int, int], torch.Tensor],
    factor_v1=1.0,
    factor_v2=1.0,
) -> dict[tuple[int, int], torch.Tensor]:
    keys = sorted(list(set(v1.keys()).union(set(v2.keys()))))
    print("Keys in the combined vector:", keys)
    combined_vector = {}
    zero_vector = None

    if v1 or v2:
        zero_vector = (
            torch.zeros_like(next(iter(v1.values())))
            if v1
            else torch.zeros_like(next(iter(v2.values())))
        )

    for layer in keys:
        vec1 = v1.get(layer, zero_vector)
        vec2 = v2.get(layer, zero_vector)
        combined_vector[layer] = factor_v1 * vec1 + factor_v2 * vec2

    return combined_vector
