from pathlib import Path
import sys
from typing import Literal
import torch
import pandas as pd
from datasets import load_dataset
import json

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.functions import get_top_k
from src.utils.icl import ICLDataset
from src.utils.evaluation import eval_bleu, eval_chrf
from src.utils.get_model import get_model


def get_noshot_prompt(prompt: str, source: str, target: str) -> str:
    map = {
        "eng_Latn": "English",
        "fra_Latn": "French",
        "spa_Latn": "Spanish",
        "por_Latn": "Portuguese",
        "jpn_Jpan": "Japanese",
        "zho_Hans": "Chinese",
        "hin_Deva": "Hindi",
        "arb_Arab": "Arabic",
        "rus_Cyrl": "Russian",
        "swh_Latn": "Swahili",
    }
    return f"Translate the following sentence from {map[source]} to {map[target]}:\n{prompt}"


def get_representations(
    model_name: str, langs: list[str], type: Literal["lang", "trad"]
) -> dict[tuple[str, str], torch.Tensor]:
    logs_probs_diff = {}

    for source in langs:
        for target in langs:
            if source == target:
                continue

            if Path(
                f"logprobs_diff_{type}/{model_name.split('/')[-1]}:{source}:{target}.pt"
            ).exists():
                log_probs_diff = torch.load(
                    f"logprobs_diff_{type}/{model_name.split('/')[-1]}:{source}:{target}.pt",
                    map_location="cpu",
                )
                logs_probs_diff[(source, target)] = log_probs_diff

    return logs_probs_diff


def add_vectors(
    v1: dict[tuple[int, int], torch.Tensor],
    v2: dict[tuple[int, int], torch.Tensor],
    factor_v1=1.0,
    factor_v2=1.0,
) -> dict[tuple[int, int], torch.Tensor]:
    keys = sorted(list(set(v1.keys()).union(set(v2.keys()))))
    print("Keys in the combined vector:", keys)
    combined_vector = {}

    for layer in keys:
        vec1 = v1.get(layer, torch.zeros_like(next(iter(v1.values()))))
        vec2 = v2.get(layer, torch.zeros_like(next(iter(v2.values()))))
        combined_vector[layer] = factor_v1 * vec1 + factor_v2 * vec2

    return combined_vector


def main(
    model_name,
    trad_source,
    trad_target,
    lang_source,
    lang_target,
    num_trad_heads,
    num_lang_heads,
):
    lang_pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (x["sentence_" + lang_source], x["sentence_" + lang_target])
        }
    )["pairs"]

    trad_pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (x["sentence_" + trad_source], x["sentence_" + trad_target])
        }
    )["pairs"]

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
    ]

    ds_lang = ICLDataset(lang_pairs, bidirectional=False)
    ds_trad = ICLDataset(trad_pairs, bidirectional=False)

    df_lang = ds_lang.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
        local_corruption=False,
    )

    df_trad = ds_trad.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
        local_corruption=False,
    )

    model = get_model(model_name)

    logprobs_diff_langs = get_representations(model_name, langs, type="lang")

    logprobs_diff_trads = get_representations(model_name, langs, type="trad")

    selected_heads_lang = get_top_k(logprobs_diff_langs.mean(dim=-1), num_lang_heads)
    selected_heads_trad = get_top_k(logprobs_diff_trads.mean(dim=-1), num_trad_heads)

    print("Selected heads (language):", selected_heads_lang)
    print("Selected heads (translation):", selected_heads_trad)

    h_lang = model.calculate_fn_vector(
        df_lang["context"].tolist(), selected_heads_lang, batch_size=64
    )

    h_trad = model.calculate_fn_vector(
        df_trad["context"].tolist(), selected_heads_trad, batch_size=64
    )

    h = add_vectors(h_lang, h_trad, factor_v1=1.0, factor_v2=1.0)

    generations_function_vector = model.generate_with_fn_vector(
        df_lang["noshot_prompt"].tolist(),
        fn_vector=h,
        max_new_tokens=75,
        stops=["\n"],
    )

    # if baseline generations do not exist, compute and save them
    if not Path(
        f"results/translation_task/generation/baseline:{model_name.split('/')[-1]}:{lang_source}:{lang_target}.csv"
    ).exists():
        generation_baseline_answer = model.generate(
            df_lang["noshot_prompt"]
            .apply(lambda x: get_noshot_prompt(x, lang_source, lang_target))
            .tolist(),
            max_new_tokens=75,
            stops=["\n"],
        )

        with open(
            f"results/translation_task/generation/baseline:{model_name.split('/')[-1]}:{lang_source}:{lang_target}.csv",
            "w",
        ) as f:
            pd.DataFrame({"generation_baseline": generation_baseline_answer}).to_csv(
                f, index=False
            )

    generation_baseline = pd.read_csv(
        f"results/translation_task/generation/baseline:{model_name.split('/')[-1]}:{lang_source}:{lang_target}.csv"
    )["generation_baseline"].tolist()

    results_df = pd.DataFrame(
        {
            "prompt": df_lang["noshot_prompt"],
            "reference": df_lang["noshot_answers"],
            "generation_baseline": generation_baseline["generation_baseline"],
            "generation_function_vector": generations_function_vector,
        }
    )

    results_df["bleu_baseline"] = results_df.apply(
        lambda row: eval_bleu(
            reference=row["reference"],
            generation=row["generation_baseline"],
        ),
        axis=1,
    )

    results_df["bleu_function_vector"] = results_df.apply(
        lambda row: eval_bleu(
            reference=row["reference"],
            generation=row["generation_function_vector"],
        ),
        axis=1,
    )

    results_df["chrf_baseline"] = results_df.apply(
        lambda row: eval_chrf(
            reference=row["reference"],
            generation=row["generation_baseline"],
        ),
        axis=1,
    )

    results_df["chrf_function_vector"] = results_df.apply(
        lambda row: eval_chrf(
            reference=row["reference"],
            generation=row["generation_function_vector"],
        ),
        axis=1,
    )

    results_df.to_csv(
        f"generation/{model_name.split('/')[-1]}:{trad_source}:{trad_target}:{lang_source}:{lang_target}:{num_trad_heads}:{num_lang_heads}.csv",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            "Usage: python generate.py <model_name> <lang_source> <lang_target> <number of trad_heads> <number of lang_heads>"
        )
        sys.exit(1)

    (
        model_name,
        trad_source,
        trad_target,
        lang_source,
        lang_target,
        num_trad_heads,
        num_lang_heads,
    ) = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        int(sys.argv[6]),
        int(sys.argv[7]),
    )

    main(
        model_name,
        trad_source,
        trad_target,
        lang_source,
        lang_target,
        num_trad_heads,
        num_lang_heads,
    )
