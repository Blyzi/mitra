from pathlib import Path
import sys
from datasets import load_dataset
import pandas as pd
import random

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.evaluation import eval_bleu, eval_chrf
from src.utils.get_model import get_model


def build_few_shot_prompt(few_shot_pairs, test_pairs, n_shots):
    few_shots = []

    for test_pair in test_pairs:
        prompt = ""
        # Select n random examples from few_shot_pairs
        selected_examples = random.sample(few_shot_pairs, n_shots)
        for i in range(n_shots):
            src, tgt = selected_examples[i]
            prompt += "Q:{src}\nA:{tgt}\n\n".format(src=src, tgt=tgt)
        src_test, _ = test_pair
        prompt += "Q:{src_test}\nA:".format(src_test=src_test)

        few_shots.append(prompt)

    return few_shots

def main(model_name: str, lang_source: str, lang_target: str, n_shots: int):
    if Path(f"results/translation_task_nshot/baseline_few_shot/{model_name.split('/')[-1]}:{lang_source}:{lang_target}:{n_shots}.csv").exists():
        print("Results already exist. Exiting.")
        return

    test_pairs = load_dataset("facebook/flores", "all")["devtest"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + lang_target],
            )
        }
    )["pairs"]

    few_shot_pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {
            "pairs": (
                " " + x["sentence_" + lang_source],
                " " + x["sentence_" + lang_target],
            )
        }
    )["pairs"]

    few_shot_prompts = build_few_shot_prompt(few_shot_pairs, test_pairs, n_shots)

    model = get_model(model_name)


    generations = model.generate(few_shot_prompts, batch_size=128, stops=["\n", "\n\n", "<eos>", "<|endoftext|>", "<|end_of_text|>"])

    df = pd.DataFrame(
        {
            "prompt": few_shot_prompts,
            "query": [pair[0] for pair in test_pairs],
            "reference": [pair[1] for pair in test_pairs],
            "generation_baseline": generations,
        }
    )

    df["bleu_baseline"] = df.apply(
        lambda row: eval_bleu(
            reference=row["reference"].strip(),
            generation=row["generation_baseline"].strip(),
        ),
        axis=1,
    )

    df["chrf_baseline"] = df.apply(
        lambda row: eval_chrf(
            reference=row["reference"].strip(),
            generation=row["generation_baseline"].strip(),
        ),
        axis=1,
    )

    Path("results/translation_task_nshot/baseline_few_shot").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"results/translation_task_nshot/baseline_few_shot/{model_name.split('/')[-1]}:{lang_source}:{lang_target}:{n_shots}.csv", index=False)

    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python metricx_evaluation.py <model_name> <lang_source> <lang_target> <number of shots>")
        sys.exit(1)

    

    model_name, lang_source, lang_target, n_shots = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    main(model_name, lang_source, lang_target, n_shots)