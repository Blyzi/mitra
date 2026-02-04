from datasets import load_dataset, Dataset
from pathlib import Path
from typing import Literal
import torch
import random
import sys

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.get_model import get_model


def get_ranked_heads(
    log_probs_diff: dict[tuple[str, str], torch.Tensor],
) -> list[tuple[str, str]]:
    """
    Rank the index of the heads based on their average log probabilities difference.
    Return the index of the head with the highest average log probabilities difference.
    """

    log_prob_diff = {
        k: v.mean(dim=-1) for k, v in log_probs_diff.items()
    }  # Average over the dataset

    num_layers, num_heads = (
        log_prob_diff[next(iter(log_prob_diff))].shape[0],
        log_prob_diff[next(iter(log_prob_diff))].shape[1],
    )

    # Initialize a dictionary to store the number of times each head is ranked first
    ranked_heads = {
        (layer, head): 0 for layer in range(num_layers) for head in range(num_heads)
    }

    for log_probs in log_prob_diff.values():
        max_head_indices = log_probs.argmax().item()

        max_layer = max_head_indices // num_heads
        max_head = max_head_indices % num_heads

        ranked_heads[(max_layer, max_head)] += 1

    # Sort the heads based on the number of times they were ranked first
    sorted_ranked_heads = sorted(ranked_heads.items(), key=lambda x: x[1], reverse=True)

    return [head for head, _ in sorted_ranked_heads]


def get_logprobs_diff(
    model_name: str, langs: list[str], type: Literal["lang", "trad"]
) -> dict[tuple[str, str], torch.Tensor]:
    logs_probs_diff = {}

    for source in langs:
        for target in langs:
            if source == target:
                continue

            if Path(
                f"results/translation_task/logprobs_diff_{type}/{model_name.split('/')[-1]}:{source}:{target}.pt"
            ).exists():
                log_probs_diff = torch.load(
                    f"results/translation_task/logprobs_diff_{type}/{model_name.split('/')[-1]}:{source}:{target}.pt",
                    map_location="cpu",
                )
                logs_probs_diff[(source, target)] = log_probs_diff

    return logs_probs_diff


def split_text_randomly(sample):
    words = sample["text"].split()

    split_index = random.randint(50, len(words) - 1)

    sample["context"] = " ".join(words[:split_index])
    sample["context_answers"] = " " + " ".join(words[split_index:])

    return sample


def main(
    model_name: str,
    type_task: str,
):
    langs = [
        "eng_Latn",
        "fra_Latn",
        "spa_Latn",
        "por_Latn",
        "jpn_Jpan",
        "zho_Hans",
        "hin_Deva",
        "arb_Arab",
        "rus_Cyrl",
        "wol_Latn",
        "swh_Latn",
    ]

    if type_task == "lang":
        logprobs_diff = get_logprobs_diff(model_name, langs, type="lang")
    else:
        logprobs_diff = get_logprobs_diff(model_name, langs, type="trad")

    top_index = get_ranked_heads(logprobs_diff)[0]

    model = get_model(model_name)

    ds_stream = (
        load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "CC-MAIN-2025-26",
            split="train",
            streaming=True,
        )
        .filter(lambda x: 512 < x["token_count"] < 1024)
        .map(split_text_randomly)
    )

    ds = Dataset.from_list(list(ds_stream.take(1000)))

    head_vector = model.calculate_head_output_force(
        ds["context"],
        ds["context_answers"],
        head=top_index,
        batch_size=1,
        index=0,
    )

    Path(f"results/representation/top_heads/{type_task}").mkdir(
        parents=True, exist_ok=True
    )
    torch.save(
        head_vector,
        f"results/representation/top_heads/{type_task}/{model_name.split('/')[-1]}:random.pt",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python top_head.py <model_name> <type>")
        sys.exit(1)

    model_name, type_task = sys.argv[1:3]
    if type_task not in ["trad", "lang"]:
        print("Error: type must be 'trad' or 'lang'")
        sys.exit(1)

    main(model_name, type_task)
