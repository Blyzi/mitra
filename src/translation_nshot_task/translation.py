from pathlib import Path
import sys
import torch
from datasets import load_dataset

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.icl import ICLDataset
from src.utils.get_model import get_model
from src.utils.functions import get_top_k


def main(
    model_name: str, lang_source: str, lang_target: str, nshot: int
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
        n_shot=nshot,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    model = get_model(
        model_name,
    )


    h = model.get_activation_patch_map(
        df["context"].tolist()[:142],
        df["corrupted_context"].tolist()[:142],
        df["context_answers"].tolist()[:142],
        batch_size=1,
        selected_heads=None,
    )

    torch.save(
        h,
        f"results/translation_task_nshot/logprobs_diff_trad/{model_name.split('/')[-1]}:{lang_source}:{lang_target}:{nshot}.pt",
    )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python translation.py <model_name> <lang_source> <lang_target> <nshot>"
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))