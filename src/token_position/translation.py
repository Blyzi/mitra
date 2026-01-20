from pathlib import Path
import sys
import torch
from datasets import load_dataset

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.icl import ICLDataset
from src.utils.get_model import get_model

def main(model_name: str, lang_source: str, lang_target: str, token_selection: str | int):
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

    h = model.get_activation_patch_map_force(
        df["context"].tolist(),
        df["corrupted_context"].tolist(),
        df["context_answers"].tolist(),
        batch_size=1,
        token_selection=token_selection,
    )

    Path("results/token_position/logprobs_diff_trad").mkdir(parents=True, exist_ok=True)
    torch.save(
        h,
        f"results/token_position/logprobs_diff_trad/{model_name.split('/')[-1]}:{lang_source}:{lang_target}:{token_selection}.pt",
    )

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python lang.py <model_name> <lang_source> <lang_target> <token_selection>")
        sys.exit(1)

    model_name, lang_source, lang_target, token_selection = sys.argv[1:5]

    # Check that token_selection is full, random or an integer
    if token_selection not in ["full", "random"] and not token_selection.isdigit():
        print("token_selection must be 'full', 'random' or an integer")
        sys.exit(1)

    if token_selection.isdigit():
        token_selection = int(token_selection)

    main(model_name, lang_source, lang_target, token_selection)