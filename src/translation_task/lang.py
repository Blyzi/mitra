from pathlib import Path
import sys
from datasets import load_dataset
import random
import torch

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.icl import ICLDataset
from src.utils.get_model import get_model
from utils.functions import get_top_k


def main(model_name, lang_source, lang_target, attribution_approximation):
    fake_langs = list(
        {
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
        - {lang_target, lang_source}
    )

    pairs_fake = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (
                x["sentence_" + lang_source],
                x["sentence_" + random.choice(fake_langs)],
            )
        }
    )["pairs"]

    pairs_target = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (x["sentence_" + lang_source], x["sentence_" + lang_target])
        }
    )["pairs"]

    icl_ds_fake = ICLDataset(pairs_fake, bidirectional=False)
    icl_ds_target = ICLDataset(pairs_target, bidirectional=False)

    df_fake = icl_ds_fake.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
        local_corruption=False,
    )
    df_target = icl_ds_target.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
        local_corruption=False,
    )

    model = get_model(
        model_name,
    )

    selected_heads = None
    if attribution_approximation:
        attribution_approximation = model.get_attribution_patch_map(
            df_target["context"].tolist(),
            df_fake["context"].tolist(),
            df_target["context_answers"].tolist(),
            batch_size=16,
        )

        selected_heads = get_top_k(
            attribution_approximation.mean(dim=-1),  # mean over samples
            top_k=20,
        )

    h = model.get_activation_patch_map(
        df_target["context"].tolist(),
        df_fake["context"].tolist(),
        df_target["context_answers"].tolist(),
        batch_size=16,
        selected_heads=selected_heads,
    )

    torch.save(
        h,
        f"results/translation_task/logprobs_diff_lang/{model_name.split('/')[-1]}:{lang_source}:{lang_target}.pt",
    )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python lang.py <model_name> <lang_source> <lang_target> <attribution_approximation>"
        )
        sys.exit(1)

    model_name, lang_source, lang_target, attribution_approximation = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        bool(int(sys.argv[4])),
    )

    print(f"Model: {model_name}, Source: {lang_source}, Target: {lang_target}")
    main(model_name, lang_source, lang_target, attribution_approximation)
