import sys

import torch
import transformers
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datasets.utils.logging import disable_progress_bar

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.metricx_model import MT5ForRegression

disable_progress_bar()

metricx_tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")
metricx_model = MT5ForRegression.from_pretrained(
    "google/metricx-24-hybrid-xxl-v2p6-bfloat16",
    device_map="auto",
    torch_dtype="bfloat16",
)
metricx_model.eval()


def eval_metricx(
    source: list[str], reference: list[str], hypothesis: list[str], is_qe: bool
) -> dict:
    def _make_input(example):
        if is_qe:
            example["input"] = (
                "source: " + example["source"] + " candidate: " + example["hypothesis"]
            )
        else:
            example["input"] = (
                "source: "
                + example["source"]
                + " candidate: "
                + example["hypothesis"]
                + " reference: "
                + example["reference"]
            )
        return example

    def _tokenize(example):
        return metricx_tokenizer(
            example["input"],
            max_length=512,
            truncation=True,
            padding=False,
        )

    def _remove_eos(example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]

        return example

    ds = Dataset.from_dict(
        {
            "source": source,
            "reference": reference,
            "hypothesis": hypothesis,
        }
    )
    ds = ds.map(_make_input, batched=False)
    ds = ds.map(_tokenize, batched=False)
    ds = ds.map(_remove_eos, batched=False)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_all_columns=True,
    )

    training_args = transformers.TrainingArguments(
        per_device_eval_batch_size=512,
        dataloader_pin_memory=False,
    )

    data_collator = transformers.DataCollatorWithPadding(
        tokenizer=metricx_tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = transformers.Trainer(
        model=metricx_model,
        args=training_args,
        data_collator=data_collator,
    )
    predictions, _, _ = trainer.predict(test_dataset=ds)

    return predictions


def generation(path: Path, model: str, is_qe: bool):
    for file in tqdm(
        path.glob(f"baseline:{model.split('/')[-1]}*.csv"),
        total=len(list(path.glob(f"baseline:{model.split('/')[-1]}*.csv"))),
        desc="Baseline",
    ):
        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if is_qe:
            column_name = "metricx_qe_baseline"
        else:
            column_name = "metricx_baseline"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_metricx(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_baseline"]],
            is_qe=is_qe,
        )

        ds = pd.read_csv(file, dtype=str, na_filter=False)
        ds[column_name] = scores
        ds.to_csv(file, index=False)

    for file in tqdm(
        path.glob(f"{model.split('/')[-1]}*.csv"),
        total=len(list(path.glob(f"{model.split('/')[-1]}*.csv"))),
        desc="Generation",
    ):
        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if is_qe:
            column_name = "metricx_qe_function_vector"
        else:
            column_name = "metricx_function_vector"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_metricx(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_function_vector"]],
            is_qe=is_qe,
        )

        ds = pd.read_csv(file, dtype=str, na_filter=False)
        ds[column_name] = scores
        ds.to_csv(file, index=False)


def ablation(model: str, is_qe: bool):
    path = Path("results/translation_task/ablation")

    for file in tqdm(
        path.glob(f"{model.split('/')[-1]}*.csv"),
        desc="Ablation",
        total=len(list(path.glob(f"{model.split('/')[-1]}*.csv"))),
    ):
        if "baseline:" in file.name:
            continue

        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if is_qe:
            column_name = "metricx_qe_ablation"
        else:
            column_name = "metricx_ablation"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_metricx(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_ablation"]],
            is_qe=is_qe,
        )

        ds = pd.read_csv(file, dtype=str, na_filter=False)
        ds[column_name] = scores
        ds.to_csv(file, index=False)

    for file in tqdm(
        path.glob(f"baseline:{model.split('/')[-1]}*.csv"),
        desc="Baseline",
        total=len(list(path.glob(f"baseline:{model.split('/')[-1]}*.csv"))),
    ):
        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if is_qe:
            column_name = "metricx_qe_baseline"
        else:
            column_name = "metricx_baseline"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_metricx(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_baseline"]],
            is_qe=is_qe,
        )

        ds = pd.read_csv(file, dtype=str, na_filter=False)
        ds[column_name] = scores
        ds.to_csv(file, index=False)

def baseline_few_shot(model_name: str, is_qe: bool):
    path = Path("results/translation_task_nshot/baseline_few_shot")
    
    for file in tqdm(
        path.glob(f"{model_name.split('/')[-1]}:*.csv"),
        desc="Baseline Few-Shot",
        total=len(
            list(
                path.glob(f"{model_name.split('/')[-1]}:*.csv")
            )
        ),
    ):
        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if is_qe:
            column_name = "metricx_qe_baseline"
        else:
            column_name = "metricx_baseline"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_metricx(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_baseline"]],
            is_qe=is_qe,
        )

        ds = pd.read_csv(file, dtype=str, na_filter=False)
        ds[column_name] = scores
        ds.to_csv(file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python metricx.py <model> <generation|ablation|amplification_factor> <is_qe>")
        sys.exit(1)

    model = sys.argv[1]
    folder = sys.argv[2]
    is_qe = sys.argv[3].lower() == "true"

    if folder == "generation":
        path = Path("results/translation_task/generation")
        generation(path, model, is_qe)
    elif folder == "amplification_factor":
        path = Path("results/amplification_factor")
        generation(path, model, is_qe)
    elif folder == "token_position":
        path = Path("results/token_position/generation")
        generation(path, model, is_qe)
    elif folder == "baseline_few_shot":
        baseline_few_shot(model, is_qe)
    elif folder == "ablation":
        ablation(model, is_qe)
