from pathlib import Path
import sys
from datasets import load_dataset
import random
import torch

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.icl import ICLDataset
from src.utils.get_model import get_model


def main(model_name, lang_source, lang_target, nshot: int):
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
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + random.choice(fake_langs)],
            )
        }
    )["pairs"]

    pairs_target = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + lang_target],
            )
        }
    )["pairs"]

    icl_ds_fake = ICLDataset(pairs_fake, bidirectional=False)
    icl_ds_target = ICLDataset(pairs_target, bidirectional=False)

    df_fake = icl_ds_fake.get_prompts(
        n_shot=nshot,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )
    df_target = icl_ds_target.get_prompts(
        n_shot=nshot,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    model = get_model(
        model_name,
    )

    h = model.get_activation_patch_map(
        df_target["context"].tolist()[:200],
        df_fake["context"].tolist()[:200],
        df_target["context_answers"].tolist()[:200],
        batch_size=1,
        selected_heads=None,
    )

    Path("results/translation_task_nshot/logprobs_diff_lang").mkdir(parents=True, exist_ok=True)
    torch.save(
        h,
        f"results/translation_task_nshot/logprobs_diff_lang/{model_name.split('/')[-1]}:{lang_source}:{lang_target}:{nshot}.pt",
    )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python lang.py <model_name> <lang_source> <lang_target> <nshot>"
        )
        sys.exit(1)

    model_name, lang_source, lang_target, nshot = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(int(sys.argv[4])),
    )

    print(f"Model: {model_name}, Source: {lang_source}, Target: {lang_target} Nshot: {nshot}")
    main(model_name, lang_source, lang_target, nshot)