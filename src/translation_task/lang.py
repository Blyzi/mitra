import torch
from src.utils.icl import ICLDataset
from src.utils.get_model import get_model
from datasets import load_dataset
import sys
import random


def main(model_name, lang_source, lang_target):
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

    h = model.get_fv_impact(
        df_target["context"].tolist(),
        df_fake["context"].tolist(),
        df_target["context_answers"].tolist(),
        batch_size=48,
    )

    torch.save(
        h,
        f"logprobs_diff_lang/{model_name.split('/')[-1]}:{lang_source}:{lang_target}.pt",
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python lang.py <model_name> <lang_source> <lang_target>")
        sys.exit(1)

    model_name, lang_source, lang_target = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
    )

    print(f"Model: {model_name}, Source: {lang_source}, Target: {lang_target}")
    main(model_name, lang_source, lang_target)
