import sys
from datasets import load_dataset
from pathlib import Path
from typing import Literal
import torch
import random
import pandas as pd


sys.path.insert(0, Path.cwd().as_posix())

from src.utils.icl import ICLDataset
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


def main(
    model_name: str,
    type_task: str,
    src_lang1: str,
    tgt_lang1: str,
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

    fake_langs = {
        "eng_Latn",
        "fra_Latn",
        "spa_Latn",
        "por_Latn",
        "jpn_Jpan",
        "zho_Hans",
        "hin_Deva",
        "arb_Arab",
        "rus_Cyrl",
    }

    if type_task == "lang":
        logprobs_diff = get_logprobs_diff(model_name, langs, type="lang")
    else:
        logprobs_diff = get_logprobs_diff(model_name, langs, type="trad")

    top_index = get_ranked_heads(logprobs_diff)[0]

    model = get_model(model_name)

    pairs1 = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + src_lang1],
                " " + x["sentence_" + tgt_lang1],
            )
        }
    )["pairs"]

    if type_task == "lang":
        pairs1_fake = load_dataset("facebook/flores", "all")["dev"].map(
            lambda x: {
                "pairs": (
                    " " + x["sentence_" + src_lang1],
                    " "
                    + x[
                        "sentence_"
                        + random.choice(list(fake_langs - {src_lang1, tgt_lang1}))
                    ],
                )
            }
        )["pairs"]

    icl1 = ICLDataset(pairs1, bidirectional=False)

    if type_task == "lang":
        icl1_fake = ICLDataset(pairs1_fake, bidirectional=False)

    df1 = icl1.get_prompts(
        n_shot=5,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    # Compute the funtion vector of the top head
    if type_task == "lang":
        df1_fake = icl1_fake.get_prompts(
            n_shot=5,
            n_shot_format="Q:{x}\nA:{y}\n\n",
            question_format="Q:{x}\nA:",
            local_corruption=False,
        )

        steering_vector = model.decode_task_vector(
            df1["context"].tolist(),
            df1_fake["context"].tolist(),
            df1["context_answers"].tolist(),
            head=top_index,
            batch_size=1,
        ).cpu()

    else:
        steering_vector = model.decode_task_vector(
            df1["context"].tolist(),
            df1["corrupted_context"].tolist(),
            df1["context_answers"].tolist(),
            head=top_index,
            batch_size=1,
        ).cpu()

    # Print the top token from the steering vector
    top_tokens = steering_vector.topk(100)

    # Save the top tokens to a csv file
    output_path = Path("results/representation/top_heads/decode")
    output_path.mkdir(parents=True, exist_ok=True)

    ds = pd.DataFrame(
        {
            "token_id": top_tokens.indices.squeeze().tolist(),
            "token": [
                model.tokenizer.decode([token_id])
                for token_id in top_tokens.indices.tolist()
            ],
        }
    )

    ds.to_csv(
        output_path
        / f"top_head_decode_{type_task}_{model_name.split('/')[-1]}_{src_lang1}_{tgt_lang1}.csv",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python top_head.py <model_name> <type> <source_language1> <target_language1>"
        )
        sys.exit(1)

    model_name, type_task, src_lang1, tgt_lang1 = sys.argv[1:5]
    if type_task not in ["trad", "lang"]:
        print("Error: type must be 'trad' or 'lang'")
        sys.exit(1)

    main(model_name, type_task, src_lang1, tgt_lang1)
