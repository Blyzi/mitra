import torch
from src.utils.icl import ICLDataset
from src.utils.get_model import get_model
from datasets import load_dataset
import sys


def main():
    model_name, lang_source, lang_target = sys.argv[1], sys.argv[2], sys.argv[3]

    pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (x["sentence_" + lang_source], x["sentence_" + lang_target])
        }
    )["pairs"]

    icl_ds = ICLDataset(pairs, bidirectional=False)

    df = icl_ds.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
        local_corruption=False,
    )

    model = get_model(
        model_name,
    )

    h = model.get_fv_impact(
        df["context"].tolist(),
        df["corrupted_context"].tolist(),
        df["context_answers"].tolist(),
        batch_size=48,
    )

    torch.save(
        h,
        f"logprobs_diff_trad/{model_name.split('/')[-1]}:{lang_source}:{lang_target}.pt",
    )


if __name__ == "__main__":
    main()
