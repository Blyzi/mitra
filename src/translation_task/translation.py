from pathlib import Path
import sys
import torch
from datasets import load_dataset

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.icl import ICLDataset
from src.utils.get_model import get_model
from src.utils.functions import get_top_k


def main(
    model_name: str, lang_source: str, lang_target: str, attribution_approximation: bool
):
    pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + lang_target],
            )
        }
    )["pairs"]
    # We add a space at the beginning to not have to add it later in the get_prompts function

    icl_ds = ICLDataset(pairs, bidirectional=False)

    df = icl_ds.get_prompts(
        n_shot=5,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    model = get_model(
        model_name,
    )

    selected_heads = None
    if attribution_approximation:
        attribution_approximation = model.get_attribution_patch_map(
            df["context"].tolist(),
            df["corrupted_context"].tolist(),
            df["context_answers"].tolist(),
            batch_size=1,
        )

        selected_heads = get_top_k(
            attribution_approximation,
            top_k=20,
        )

    h = model.get_activation_patch_map(
        df["context"].tolist(),
        df["corrupted_context"].tolist(),
        df["context_answers"].tolist(),
        batch_size=1,
        selected_heads=selected_heads,
    )

    torch.save(
        h,
        f"results/translation_task/logprobs_diff_trad/{model_name.split('/')[-1]}:{lang_source}:{lang_target}.pt",
    )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python translation.py <model_name> <lang_source> <lang_target> <attribution_approximation>"
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4])))
