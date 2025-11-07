from pathlib import Path
import sys
from typing import Literal
import torch
from src.utils.icl import ICLDataset
from src.utils.evaluation import eval_bleu, eval_chrf
from src.utils.get_model import get_model
from datasets import load_dataset
from src.utils.functions import get_logprobs_diff_elbow, get_top_k
import pandas as pd


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


def get_selected_heads(
    log_probs_diff: dict[tuple[str, str], torch.Tensor],
) -> list[tuple[int, int]]:
    # Get the top k heads based on the elbow method

    k = get_logprobs_diff_elbow(log_probs_diff, list(log_probs_diff.keys()))

    top_heads = get_top_k(
        torch.mean(torch.stack(list(log_probs_diff.values())), dim=0), k
    )

    return sorted(top_heads, key=lambda x: x[0])


def main(model_name, lang_source, lang_target):
    pairs_source_target = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (x["sentence_" + lang_source], x["sentence_" + lang_target])
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

    icl_ds = ICLDataset(pairs_source_target, bidirectional=False)

    df = icl_ds.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
        local_corruption=False,
    )

    model = get_model(model_name)

    logprobs_diff_langs = get_representations(model_name, langs, type="lang")

    logprobs_diff_trads = get_representations(model_name, langs, type="trad")

    selected_heads_lang = get_selected_heads(logprobs_diff_langs)
    selected_heads_trad = get_selected_heads(logprobs_diff_trads)

    print("Selected heads (language):", selected_heads_lang)
    print("Selected heads (translation):", selected_heads_trad)

    h_lang = model.calculate_fn_vector(
        df["context"].tolist(), selected_heads_lang, batch_size=64
    )
    h_trad = model.calculate_fn_vector(
        df["context"].tolist(), selected_heads_trad, batch_size=64
    )

    h = add_vectors(h_lang, h_trad, factor_v1=0.0, factor_v2=6.0)

    generations__function_vector = model.generate_with_fn_vector(
        df["noshot_prompt"].tolist(),
        fn_vector=h,
        max_new_tokens=50,
        stops=["\n"],
    )

    generation_baseline = model.generate(
        df["noshot_prompt"]
        .apply(lambda x: get_noshot_prompt(x, lang_source, lang_target))
        .tolist(),
        max_new_tokens=50,
        stops=["\n"],
    )

    results_df = pd.DataFrame(
        {
            "prompt": df["noshot_prompt"],
            "reference": df["noshot_answers"],
            "generation_baseline": generation_baseline,
            "generation_function_vector": generations__function_vector,
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
        f"generation/{model_name.split('/')[-1]}:{lang_source}:{lang_target}.csv",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate.py <model_name> <lang_source> <lang_target>")
        sys.exit(1)

    model_name, lang_source, lang_target = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
    )

    main(model_name, lang_source, lang_target)
