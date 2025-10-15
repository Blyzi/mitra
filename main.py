from utils.data import ICLDataset
from utils.models import Model
from datasets import load_dataset
import os
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bleu = BLEU(tokenize="flores200", effective_order=True)
chrf = CHRF(word_order=2)


def eval_bleu(reference: str, generation: str) -> float:
    return bleu.sentence_score(generation, [reference]).score


def eval_chrf(reference: str, generation: str) -> float:
    return chrf.sentence_score(generation, [reference]).score


def main():
    lang1, lang2 = "eng_Latn", "fra_Latn"

    pairs = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {"pairs": (x["sentence_" + lang1], x["sentence_" + lang2])}
    )["pairs"]

    icl_ds = ICLDataset(pairs, bidirectional=False)

    df = icl_ds.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
    )

    model = Model(
        "meta-llama/Llama-2-7b-hf",
        layers_adr=["model", "layers"],
        n_head_key="num_attention_heads",
        attn_key="self_attn",
        out_proj_key="o_proj",
        d_head_key="head_dim",
    )

    # print(model.llm.tokenizer.tokenize(df["context"].tolist()[0]))
    # return

    h = model.get_fv(
        df["context"].tolist(),
        df["corrupted_context"].tolist(),
        df["answer"].tolist(),
        icl_ds.get_token_indexes(
            df["context"].tolist(),
            token=":",
            tokenizer=model.tokenizer,
        ).tolist(),
        icl_ds.get_token_indexes(
            df["corrupted_context"].tolist(),
            token=":",
            tokenizer=model.tokenizer,
        ).tolist(),
        batch_size=64,
    )

    print(h.shape)
    print(h)

    return

    h = model.get_representations(
        df["context"].tolist(),
        icl_ds.get_token_indexes(
            df["context"].tolist(),
            token=":",
            tokenizer=model.tokenizer,
        ),
    )

    # Generate without intervention
    generation_df = model.generate(
        df["noshot_prompt"].tolist(),
        max_new_tokens=5,
        stops=["\n"],
    )

    df = pd.concat([df, generation_df], axis=1)

    df["completion_bleu"] = df.apply(
        lambda x: eval_bleu(x["answer"], x["completion"]), axis=1
    )
    df["completion_chrf"] = df.apply(
        lambda x: eval_chrf(x["answer"], x["completion"]), axis=1
    )

    # Evaluate and generate with intervention at each layer
    for layer in range(10, 11):
        print(f"Layer {layer}:")
        generation_intervention_df = model.generate_with_intervention(
            df["noshot_prompt"].tolist(),
            h,
            tokens_idx=icl_ds.get_token_indexes(
                df["noshot_prompt"].tolist(), token=":", tokenizer=model.tokenizer
            ),
            layer=layer,
            max_new_tokens=5,
            stops=["\n"],
        ).rename(
            columns={
                "completion_intervention": f"completion_intervention_{layer}",
            }
        )

        df = pd.concat([df, generation_intervention_df], axis=1)

        df[f"completion_intervention_{layer}_bleu"] = df.apply(
            lambda x: eval_bleu(x["answer"], x[f"completion_intervention_{layer}"]),
            axis=1,
        )
        df[f"completion_intervention_{layer}_chrf"] = df.apply(
            lambda x: eval_chrf(x["answer"], x[f"completion_intervention_{layer}"]),
            axis=1,
        )

    df.to_csv("results_flores_w2w.csv", index=False)

    print(
        f"  Without intervention - BLEU: {df['completion_bleu'].mean():.2f}, CHRF: {df['completion_chrf'].mean():.2f}"
    )
    # Print average scores
    for layer in range(10, 11):
        print(f"Layer {layer} results:")
        print(
            f"  With intervention - BLEU: {df[f'completion_intervention_{layer}_bleu'].mean():.2f}, CHRF: {df[f'completion_intervention_{layer}_chrf'].mean():.2f}"
        )


if __name__ == "__main__":
    main()
