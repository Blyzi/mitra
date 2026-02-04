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
    model_name: str, langs: list[str], type: Literal["lang", "trad"], token_position
) -> dict[tuple[str, str], torch.Tensor]:
    logs_probs_diff = {}

    for source in langs:
        for target in langs:
            if source == target:
                continue

            if Path(
                f"results/token_position/logprobs_diff_{type}/{model_name.split('/')[-1]}:{source}:{target}:{token_position}.pt"
            ).exists():
                log_probs_diff = torch.load(
                    f"results/token_position/logprobs_diff_{type}/{model_name.split('/')[-1]}:{source}:{target}:{token_position}.pt",
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
    token_position,
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

    test_pairs = load_dataset("facebook/flores", "all")["devtest"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + lang_target],
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
        n_shot=5,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    df_lang_fake = ds_lang_fake.get_prompts(
        n_shot=5,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    df_trad = ds_trad.get_prompts(
        n_shot=5,
        n_shot_format="Q:{x}\nA:{y}\n\n",
        question_format="Q:{x}\nA:",
        local_corruption=False,
    )

    test_queries = [pair[0].strip() for pair in test_pairs]
    test_prompts = ["Q:{x}\nA:".format(x=pair[0]) for pair in test_pairs]
    test_answers = [pair[1].strip() for pair in test_pairs]

    model = get_model(model_name)

    num_lang_heads = perc_lang_heads * round(model.n_head * model.n_layers * 0.01)
    num_trad_heads = perc_trad_heads * round(model.n_head * model.n_layers * 0.01)

    # Logs
    print("=" * 20, "Generation with function vectors", "=" * 20)
    print(f"Model: {model_name}")
    print(f"Language pair: {lang_source} -> {lang_target}")
    print(f"Translation pair: {trad_source} -> {trad_target}")
    print(f"Number of translation heads: {num_trad_heads} ({perc_trad_heads}%)")
    print(f"Number of language heads: {num_lang_heads} ({perc_lang_heads}%)")
    print("Token position:", token_position)
    print("=" * 60)

    # Check if the generation results already exist
    if Path(
        f"results/token_position/generation/{model_name.split('/')[-1]}:{trad_source}:{trad_target}:{lang_source}:{lang_target}:{num_trad_heads}:{num_lang_heads}:{token_position}.csv"
    ).exists():
        print("Generation results already exist. Exiting.")
        return

    logprobs_diff_langs = get_logprobs_diff(model_name, langs, type="lang", token_position=token_position)

    logprobs_diff_trads = get_logprobs_diff(model_name, langs, type="trad", token_position=token_position)

    selected_heads_lang = get_top_k(
        logprobs_diff_langs[(lang_source, lang_target)].mean(dim=-1), num_lang_heads
    )
    selected_heads_trad = get_top_k(
        logprobs_diff_trads[(trad_source, trad_target)].mean(dim=-1), num_trad_heads
    )

    print("Selected heads (language):", selected_heads_lang)
    print("Selected heads (translation):", selected_heads_trad)

    if len(selected_heads_lang) > 0:
        h_lang = model.calculate_fn_vector_force(
            df_lang["context"].tolist(),
            df_lang["context_answers"].tolist(),
            selected_heads_lang,
            token_position=token_position,
            batch_size=1,
        )
    else:
        h_lang = {}

    if len(selected_heads_trad) > 0:
        h_trad = model.calculate_fn_vector_force(
            df_trad["context"].tolist(),
            df_trad["context_answers"].tolist(),
            selected_heads_trad,
            token_position=token_position,
            batch_size=1,
        )
    else:
        h_trad = {}

    h = add_vectors(h_lang, h_trad, factor_v1=1.0, factor_v2=1.0)

    generations_function_vector = model.generate_with_fn_vector(
        test_prompts,
        fn_vector=h,
        max_new_tokens=100,
        stops=["\n", "\n\n", "<eos>", "<|endoftext|>", "<|end_of_text|>"],
        batch_size=2000
    )

    print("Generations with function vector done.")

    results_df = pd.DataFrame(
        {
            "prompt": test_prompts,
            "query": test_queries,
            "reference": test_answers,
            "generation_function_vector": generations_function_vector,
        },
        dtype=str,
    )

    print("Evaluating generations...")

    results_df["bleu_function_vector"] = results_df.apply(
        lambda row: eval_bleu(
            reference=row["reference"],
            generation=row["generation_function_vector"],
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

    results_df["function_vector_lang"] = results_df["generation_function_vector"].apply(
        lambda x: get_language(x),
    )

    all_answers = get_all_answers()

    for i, row in results_df.iterrows():
        lang = row["function_vector_lang"]
        if lang in all_answers:
            results_df.at[i, "reference_lang"] = all_answers[lang][i]
            results_df.at[i, "bleu_function_vector_lang"] = eval_bleu(
                reference=all_answers[lang][i],
                generation=row["generation_function_vector"],
            )
            results_df.at[i, "chrf_function_vector_lang"] = eval_chrf(
                reference=all_answers[lang][i],
                generation=row["generation_function_vector"],
            )
        else:
            results_df.at[i, "reference_lang"] = ""
            results_df.at[i, "bleu_function_vector_lang"] = None
            results_df.at[i, "chrf_function_vector_lang"] = None

    print("Saving results...")

    Path("results/token_position/generation").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        f"results/token_position/generation/{model_name.split('/')[-1]}:{trad_source}:{trad_target}:{lang_source}:{lang_target}:{num_trad_heads}:{num_lang_heads}:{token_position}.csv",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print(
            "Usage: python generate.py <model_name> <trad_source> <trad_target> <lang_source> <lang_target> <% of trad_heads> <% of lang_heads> <token_position>"
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
        token_position,
    ) = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        int(sys.argv[6]),
        int(sys.argv[7]),
        sys.argv[8],
    )

    if token_position not in ["random", "full"] and not token_position.isdigit():
        print("token_position must be 'full', 'random' or an integer")
        sys.exit(1)

    if token_position.isdigit():
        token_position = int(token_position)

    main(
        model_name,
        trad_source,
        trad_target,
        lang_source,
        lang_target,
        perc_trad_heads,
        perc_lang_heads,
        token_position,
    )
