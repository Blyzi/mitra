import sys
from pathlib import Path
from comet import download_model, load_from_checkpoint
import pandas as pd
from tqdm import tqdm

comet_model_path = download_model("Unbabel/XCOMET-XXL")
comet_model = load_from_checkpoint(comet_model_path)


def eval_commet(
    source: list[str], reference: list[str], hypothesis: list[str]
) -> list[float]:
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(source, hypothesis, reference)
    ]
    scores = comet_model.predict(data, batch_size=128, gpus=1).scores
    return scores


def generation(model: str):
    path = Path("results/translation_task/generation")

    for file in tqdm(path.glob(f"baseline:{model.split('/')[-1]}*.csv")):
        # print(f"Evaluating {file.name}...")

        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if "commet_baseline" in ds.columns:
            continue

        ds["commet_baseline"] = eval_commet(
            source=ds["query"],
            reference=ds["reference"],
            hypothesis=ds["generation_baseline"],
        )

        ds.to_csv(file, index=False)

    for file in tqdm(path.glob(f"{model.split('/')[-1]}*.csv")):
        if "baseline:" in file.name:
            continue

        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if "commet_function_vector" in ds.columns.tolist():
            continue

        ds["commet_function_vector"] = eval_commet(
            source=ds["query"],
            reference=ds["reference"],
            hypothesis=ds["generation_function_vector"],
        )

        ds.to_csv(file, index=False)


def ablation(model: str):
    path = Path("results/translation_task/ablation")

    for file in tqdm(path.glob(f"baseline:{model.split('/')[-1]}*.csv")):
        # print(f"Evaluating {file.name}...")

        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if "commet_baseline" in ds.columns.tolist():
            continue

        ds["commet_baseline"] = eval_commet(
            source=ds["query"],
            reference=ds["reference"],
            hypothesis=ds["generation_baseline"],
        )

        ds.to_csv(file, index=False)

    for file in tqdm(path.glob(f"{model.split('/')[-1]}*.csv")):
        if "baseline:" in file.name:
            continue

        ds = pd.read_csv(file, dtype=str, na_filter=False)

        if "commet_function_vector" in ds.columns:
            continue

        ds["commet_function_vector"] = eval_commet(
            source=ds["query"],
            reference=ds["reference"],
            hypothesis=ds["generation_function_vector"],
        )

        ds.to_csv(file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/evaluation.py/commet.py <model> <folder>")
        sys.exit(1)

    if sys.argv[2] not in ["generation", "ablation"]:
        print("Folder must be either 'generation' or 'ablation'")
        sys.exit(1)

    model = sys.argv[1]
    folder = sys.argv[2]

    if folder == "generation":
        generation(model)
    else:
        ablation(model)
