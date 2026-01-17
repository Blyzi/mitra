from pathlib import Path
import sys
from typing import Literal
import torch
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.functions import get_top_k
from src.utils.evaluation import eval_bleu, eval_chrf
from src.utils.get_model import get_model
from src.utils.fasttext import get_language, flores_langs


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
        "wol_Latn": "Wolof",
    }
    return f"Translate the following sentence from {map[source]} to {map[target]}:\n{prompt}"


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


def get_all_answers():
    answer_dict = {}
    ds = load_dataset("facebook/flores", "all")["devtest"]

    for lang in flores_langs:
        answer_dict[lang] = ds.map(lambda x: {"answer": " " + x["sentence_" + lang]})[
            "answer"
        ]

    return answer_dict


def main(
    model_name,
    trad_source,
    trad_target,
    lang_source,
    lang_target,
    perc_trad_heads,
    perc_lang_heads,
):
    test_pairs = load_dataset("facebook/flores", "all")["devtest"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + lang_target],
            )
        }
    )["pairs"]

    test_queries = [pair[0].strip() for pair in test_pairs]
    test_prompts = list(
        map(
            lambda x: get_noshot_prompt(x, lang_source, lang_target),
            ["Q:{x}\nA:".format(x=pair[0]) for pair in test_pairs],
        )
    )
    test_answers = [pair[1].strip() for pair in test_pairs]

    model = get_model(model_name)

    num_lang_heads = perc_lang_heads * round(model.n_head * model.n_layers * 0.01)
    num_trad_heads = perc_trad_heads * round(model.n_head * model.n_layers * 0.01)

    # Logs
    print("=" * 20, "Generation with ablation", "=" * 20)
    print(f"Model: {model_name}")
    print(f"Translation pair: {trad_source} -> {trad_target}")
    print(f"Language pair: {lang_source} -> {lang_target}")
    print(f"Number of translation heads: {num_trad_heads} ({perc_trad_heads}%)")
    print(f"Number of language heads: {num_lang_heads} ({perc_lang_heads}%)")
    print("=" * 60)

    # Check if the generation results already exist
    if Path(
        f"results/translation_task/ablation/{model_name.split('/')[-1]}:{trad_source}:{trad_target}:{lang_source}:{lang_target}:{num_trad_heads}:{num_lang_heads}.csv"
    ).exists():
        print("Generation results already exist. Exiting.")
        return

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

    logprobs_diff_langs = get_logprobs_diff(model_name, langs, type="lang")

    logprobs_diff_trads = get_logprobs_diff(model_name, langs, type="trad")

    selected_heads_lang = get_top_k(
        logprobs_diff_langs[(lang_source, lang_target)].mean(dim=-1), num_lang_heads
    )
    selected_heads_trad = get_top_k(
        logprobs_diff_trads[(trad_source, trad_target)].mean(dim=-1), num_trad_heads
    )

    print("Selected heads (language):", selected_heads_lang)
    print("Selected heads (translation):", selected_heads_trad)

    generations_ablation = model.generate_with_ablation(
        test_prompts,
        max_new_tokens=100,
        stops=["\n", "\n\n", "<eos>", "<|endoftext|>", "<|end_of_text|>"],
        heads_to_ablate=selected_heads_lang + selected_heads_trad,
    )

    print("Generations with ablation done.")

    # if baseline generations do not exist, compute and save them
    if not Path(
        f"results/translation_task/ablation/baseline:{model_name.split('/')[-1]}:{trad_source}:{trad_target}:{lang_source}:{lang_target}:{num_trad_heads + num_lang_heads}.csv"
    ).exists():
        generation_baseline_answer = model.generate_with_ablation(
            test_prompts,
            heads_to_ablate=selected_heads_lang + selected_heads_trad,
            max_new_tokens=100,
            stops=["\n", "\n\n", "<eos>", "<|endoftext|>", "<|end_of_text|>"],
            random_ablation=True,
        )

        baseline_df = pd.DataFrame(
            {
                "prompt": test_prompts,
                "query": test_queries,
                "reference": test_answers,
                "generation_baseline": generation_baseline_answer,
            }
        )

        baseline_df["chrf_baseline"] = baseline_df.apply(
            lambda row: eval_chrf(
                reference=row["reference"],
                generation=row["generation_baseline"],
            ),
            axis=1,
        )

        baseline_df["bleu_baseline"] = baseline_df.apply(
            lambda row: eval_bleu(
                reference=row["reference"],
                generation=row["generation_baseline"],
            ),
            axis=1,
        )

        Path("results/translation_task/ablation").mkdir(parents=True, exist_ok=True)
        with open(
            f"results/translation_task/ablation/baseline:{model_name.split('/')[-1]}:{trad_source}:{trad_target}:{lang_source}:{lang_target}:{num_trad_heads + num_lang_heads}.csv",
            "w",
        ) as f:
            baseline_df.to_csv(f, index=False)

    results_df = pd.DataFrame(
        {
            "prompt": test_prompts,
            "query": test_queries,
            "reference": test_answers,
            "generation_ablation": generations_ablation,
        },
        dtype=str,
    )

    print("Evaluating generations...")

    results_df["bleu_ablation"] = results_df.apply(
        lambda row: eval_bleu(
            reference=row["reference"],
            generation=row["generation_ablation"],
        ),
        axis=1,
    )

    results_df["chrf_ablation"] = results_df.apply(
        lambda row: eval_chrf(
            reference=row["reference"],
            generation=row["generation_ablation"],
        ),
        axis=1,
    )

    results_df["ablation_lang"] = results_df["generation_ablation"].apply(
        lambda x: get_language(x),
    )

    all_answers = get_all_answers()

    for i, row in results_df.iterrows():
        lang = row["ablation_lang"]
        if lang in all_answers:
            results_df.at[i, "reference_lang"] = all_answers[lang][i]
            results_df.at[i, "bleu_ablation_lang"] = eval_bleu(
                reference=all_answers[lang][i],
                generation=row["generation_ablation"],
            )
            results_df.at[i, "chrf_ablation_lang"] = eval_chrf(
                reference=all_answers[lang][i],
                generation=row["generation_ablation"],
            )
        else:
            results_df.at[i, "reference_lang"] = ""
            results_df.at[i, "bleu_ablation_lang"] = None
            results_df.at[i, "chrf_ablation_lang"] = None

    print("Saving results...")

    Path("results/translation_task/ablation").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        f"results/translation_task/ablation/{model_name.split('/')[-1]}:{trad_source}:{trad_target}:{lang_source}:{lang_target}:{num_trad_heads}:{num_lang_heads}.csv",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python generate.py <model_name> <trad_source> <trad_target> <lang_source> <lang_target> <% of trad_heads> <% of lang_heads>"
        )
        sys.exit(1)

    (
        model_name,
        trad_source,
        trad_target,
        lang_source,
        lang_target,
        perc_trad_heads,
        perc_lang_heads,
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
        perc_trad_heads,
        perc_lang_heads,
    )
