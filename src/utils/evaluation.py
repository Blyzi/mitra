from sacrebleu.metrics import BLEU, CHRF
import transformers
from datasets import Dataset

bleu = BLEU(tokenize="flores200", effective_order=True)
chrf = CHRF(word_order=2)


def eval_bleu(reference: str, generation: str) -> float:
    return bleu.sentence_score(generation, [reference]).score


def eval_chrf(reference: str, generation: str) -> float:
    return chrf.sentence_score(generation, [reference]).score


def eval_metricx(
    source: list[str], reference: list[str], hypothesis: list[str], is_qe: bool
) -> dict:
    from src.utils.metricx_model import MT5ForRegression

    metricx_tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")
    metricx_model = MT5ForRegression.from_pretrained(
        "google/metricx-24-hybrid-xxl-v2p6-bfloat16"
    )

    def _make_input(example, is_qe=is_qe):
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
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_all_columns=True,
    )

    training_args = transformers.TrainingArguments(
        per_device_eval_batch_size=1,
        dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(
        model=metricx_model,
        args=training_args,
    )
    predictions, _, _ = trainer.predict(test_dataset=ds)

    del metricx_model

    return predictions


def eval_comet(
    source: list[str], reference: list[str], hypothesis: list[str]
) -> list[float]:
    from comet import download_model, load_from_checkpoint

    comet_model_path = download_model("Unbabel/XCOMET-XXL")
    comet_model = load_from_checkpoint(comet_model_path)

    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(source, hypothesis, reference)
    ]

    res = comet_model.predict(data, batch_size=1, gpus=1)

    del comet_model

    return res.scores
