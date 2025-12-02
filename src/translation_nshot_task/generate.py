from pathlib import Path
import random
import sys
from typing import Literal
import torch
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.functions import get_top_k
from src.utils.icl import ICLDataset
from src.utils.evaluation import eval_bleu, eval_chrf
from src.utils.get_model import get_model
from src.utils.fasttext import get_language, flores_langs
from src.utils.add_vectors import add_vectors


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
                f"results/translation_task_nshot/logprobs_diff_{type}/{model_name.split('/')[-1]}:{source}:{target}.pt"
            ).exists():
                log_probs_diff = torch.load(
                    f"results/translation_task_nshot/logprobs_diff_{type}/{model_name.split('/')[-1]}:{source}:{target}.pt",
                    map_location="cpu",
                )
                logs_probs_diff[(source, target)] = log_probs_diff

    return logs_probs_diff


def get_all_answers(nshot: int = 5) -> dict[str, list[str]]:
    answer_dict = {}
    ds = load_dataset("facebook/flores", "all")["dev"]

    for lang in flores_langs:
        pairs = ds.map(
            lambda x: {
                "pairs": (
                    "",
                    " " + x["sentence_" + lang],
                )
            }
        )["pairs"]

        icl = ICLDataset(pairs, bidirectional=False)

        df = icl.get_prompts(
            n_shot=nshot,
            n_shot_format="Q:{x}\nA:{y}\n\n",
            question_format="Q:{x}\nA:",
            local_corruption=False,
        )

        answer_dict[lang] = df["noshot_answers"].tolist()[:142]

    return answer_dict


def main(
    model_name,
    trad_source,
    trad_target,
    lang_source,
    lang_target,
    num_trad_heads,
    num_lang_heads,
    nshot,
):
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

    lang_pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + lang_target],
            )
        }
    )["pairs"]

    lang_fake_pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + random.choice(fake_langs)],
            )
        }
    )["pairs"]

    trad_pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + trad_source],
                " " + x["sentence_" + trad_target],
            )
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
        "wol_Latn",
        "swh_Latn",
    ]

    ds_lang = ICLDataset(lang_pairs, bidirectional=False)
    ds_lang_fake = ICLDataset(lang_fake_pairs, bidirectional=False)
    ds_trad = ICLDataset(trad_pairs, bidirectional=False)

    df_lang = ds_lang.get_prompts(
        n_shot=nshot,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    df_lang_fake = ds_lang_fake.get_prompts(
        n_shot=nshot,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    df_trad = ds_trad.get_prompts(
        n_shot=nshot,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    model = get_model(model_name)

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

    if len(selected_heads_lang) > 0:
        h_lang = model.calculate_fn_vector(
            df_lang["context"].tolist()[:142],
            df_lang_fake["context"].tolist()[:142],
            df_lang["context_answers"].tolist()[:142],
            selected_heads_lang,
            batch_size=1,
        )
    else:
        h_lang = {}

    if len(selected_heads_trad) > 0:
        h_trad = model.calculate_fn_vector(
            df_trad["context"].tolist()[:142],
            df_trad["corrupted_context"].tolist()[:142],
            df_trad["context_answers"].tolist()[:142],
            selected_heads_trad,
            batch_size=1,
        )
    else:
        h_trad = {}

    h = add_vectors(h_lang, h_trad, factor_v1=1.0, factor_v2=1.0)

    generations_function_vector = model.generate_with_fn_vector(
        df_lang["noshot_prompt"].tolist(),
        fn_vector=h,
        max_new_tokens=75,
        stops=["\n", "<eos>", "<|endoftext|>"],
    )

    print("Generations with function vector done.")

    # if baseline generations do not exist, compute and save them
    if not Path(
        f"results/translation_task_nshot/generation/baseline:{model_name.split('/')[-1]}:{lang_source}:{lang_target}.csv"
    ).exists():
        generation_baseline_answer = model.generate(
            df_lang["noshot_prompt"]
            .apply(lambda x: get_noshot_prompt(x, lang_source, lang_target))
            .tolist(),
            max_new_tokens=75,
            stops=["\n", "<eos>", "<|endoftext|>"],
        )

        with open(
            f"results/translation_task_nshot/generation/baseline:{model_name.split('/')[-1]}:{lang_source}:{lang_target}.csv",
            "w",
        ) as f:
            pd.DataFrame({"generation_baseline": generation_baseline_answer}).to_csv(
                f, index=False
            )

    generation_baseline = pd.read_csv(
        f"results/translation_task_nshot/generation/baseline:{model_name.split('/')[-1]}:{lang_source}:{lang_target}.csv",
        dtype=str,
        na_filter=False,
    )["generation_baseline"].tolist()

    results_df = pd.DataFrame(
        {
            "prompt": df_lang["noshot_prompt"],
            "reference": df_lang["noshot_answers"],
            "generation_baseline": generation_baseline,
            "generation_function_vector": generations_function_vector,
        },
        dtype=str,
    )

    print("Evaluating generations...")

    results_df["bleu_baseline"] = results_df.apply(
        lambda row: eval_bleu(
            reference=row["reference"].strip(),
            generation=row["generation_baseline"].strip(),
        ),
        axis=1,
    )

    results_df["bleu_function_vector"] = results_df.apply(
        lambda row: eval_bleu(
            reference=row["reference"].strip(),
            generation=row["generation_function_vector"].strip(),
        ),
        axis=1,
    )

    results_df["chrf_baseline"] = results_df.apply(
        lambda row: eval_chrf(
            reference=row["reference"].strip(),
            generation=row["generation_baseline"].strip(),
        ),
        axis=1,
    )

    results_df["chrf_function_vector"] = results_df.apply(
        lambda row: eval_chrf(
            reference=row["reference"].strip(),
            generation=row["generation_function_vector"].strip(),
        ),
        axis=1,
    )

    results_df["function_vector_lang"] = results_df["generation_function_vector"].apply(
        lambda x: get_language(x),
    )

    all_answers = get_all_answers()

    for i, row in results_df.iterrows():
        lang = row["function_vector_lang"]
        if lang in all_answers:
            results_df.at[i, "bleu_function_vector_lang"] = eval_bleu(
                reference=all_answers[lang][i].strip(),
                generation=row["generation_function_vector"].strip(),
            )
            results_df.at[i, "chrf_function_vector_lang"] = eval_chrf(
                reference=all_answers[lang][i].strip(),
                generation=row["generation_function_vector"].strip(),
            )
        else:
            results_df.at[i, "bleu_function_vector_lang"] = None
            results_df.at[i, "chrf_function_vector_lang"] = None

    print("Saving results...")

    results_df.to_csv(
        f"results/translation_task_nshot/generation/{model_name.split('/')[-1]}:{trad_source}:{trad_target}:{lang_source}:{lang_target}:{num_trad_heads}:{num_lang_heads}.csv",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print(
            "Usage: python generate.py <model_name> <trad_source> <trad_target> <lang_source> <lang_target> <number of trad_heads> <number of lang_heads> <nshot>"
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
        nshot,
    ) = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        int(sys.argv[6]),
        int(sys.argv[7]),
        int(sys.argv[8]),
    )

    main(
        model_name,
        trad_source,
        trad_target,
        lang_source,
        lang_target,
        num_trad_heads,
        num_lang_heads,
        nshot,
    )
