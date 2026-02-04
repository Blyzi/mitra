from functools import partial
import sys
from pathlib import Path
from comet import download_model, load_from_checkpoint
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler

comet_model_path = download_model("Unbabel/XCOMET-XXL")
comet_model = load_from_checkpoint(comet_model_path)

comet_model = comet_model.to(device="cuda", dtype=torch.bfloat16)
comet_model.eval()


def eval_commet(
    source: list[str], reference: list[str], hypothesis: list[str]
) -> list[float]:
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(source, hypothesis, reference)
    ]

    dataloader = DataLoader(
        dataset=data,
        batch_size=128,
        sampler=SequentialSampler(data),
        collate_fn=comet_model.prepare_for_inference,
        num_workers=0,
        shuffle=False,
    )
    
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            model_inputs = batch[0]
            
            # Move tensors to GPU, keep non-tensors (like mt_offsets) as-is
            model_inputs_gpu = {}
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs_gpu[k] = v.to(comet_model.device)
                else:
                    model_inputs_gpu[k] = v
            
            # Forward pass
            model_output = comet_model.forward(**model_inputs_gpu)
            scores = model_output.score.cpu().tolist()
            all_scores.extend(scores)

    return all_scores

def generation(path: Path, model: str):
    for file in tqdm(
        path.glob(f"baseline:{model.split('/')[-1]}*.csv"),
        total=len(list(path.glob(f"baseline:{model.split('/')[-1]}*.csv"))),
        desc="Baseline",
    ):
        ds = pd.read_csv(file, dtype=str, na_filter=False)

        column_name = "commet_baseline"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_commet(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_baseline"]],
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

        column_name = "commet_function_vector"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_commet(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_function_vector"]],
        )

        ds = pd.read_csv(file, dtype=str, na_filter=False)
        ds[column_name] = scores
        ds.to_csv(file, index=False)


def ablation(model: str):
    path = Path("results/translation_task/ablation")

    for file in tqdm(
        path.glob(f"{model.split('/')[-1]}*.csv"),
        desc="Ablation",
        total=len(list(path.glob(f"{model.split('/')[-1]}*.csv"))),
    ):
        if "baseline:" in file.name:
            continue

        ds = pd.read_csv(file, dtype=str, na_filter=False)

        column_name = "commet_ablation"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_commet(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_ablation"]],
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

        column_name = "commet_baseline"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_commet(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_baseline"]],
        )

        ds = pd.read_csv(file, dtype=str, na_filter=False)
        ds[column_name] = scores
        ds.to_csv(file, index=False)

def baseline_few_shot(model_name: str):
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

        column_name = "commet_baseline"

        if column_name in ds.columns.tolist():
            continue

        scores = eval_commet(
            source=[d for d in ds["query"]],
            reference=[d for d in ds["reference"]],
            hypothesis=[d for d in ds["generation_baseline"]],
        )

        ds = pd.read_csv(file, dtype=str, na_filter=False)
        ds[column_name] = scores
        ds.to_csv(file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python commet.py <model> <generation|ablation|amplification_factor>")
        sys.exit(1)

    model = sys.argv[1]
    folder = sys.argv[2]

    if folder == "generation":
        path = Path("results/translation_task/generation")
        generation(path, model)
    elif folder == "amplification_factor":
        path = Path("results/amplification_factor")
        generation(path, model)
    elif folder == "token_position":
        path = Path("results/token_position/generation")
        generation(path, model)
    elif folder == "baseline_few_shot":
        baseline_few_shot(model)
    elif folder == "ablation":
        ablation(model)