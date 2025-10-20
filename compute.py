import torch
from utils.data import ICLDataset
from utils.models import Model
from datasets import load_dataset
import os
import pandas as pd
from utils.evaluation import eval_bleu, eval_chrf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    lang_en, lang_fr, lang_pt = "eng_Latn", "fra_Latn", "por_Latn"

    pairs_en_fr = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {"pairs": (x["sentence_" + lang_en], x["sentence_" + lang_fr])}
    )["pairs"]
    pairs_en_pt = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {"pairs": (x["sentence_" + lang_en], x["sentence_" + lang_pt])}
    )["pairs"]

    icl_ds_en_fr = ICLDataset(pairs_en_fr, bidirectional=False)
    icl_ds_en_pt = ICLDataset(pairs_en_pt, bidirectional=False)

    df_en_fr = icl_ds_en_fr.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
        local_corruption=False,
    )
    df_en_pt = icl_ds_en_pt.get_prompts(
        n_shot=5,
        n_shot_format="Q: {x}\nA: {y}\n\n",
        question_format="Q: {x}\nA:",
        local_corruption=False,
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

    print("Computing logprobs difference for EN-FR...")
    print(df_en_fr["context"].tolist()[0])
    print(df_en_fr["corrupted_context"].tolist()[0])
    print(df_en_fr["context_answers"].tolist()[0])

    h = model.get_fv_impact(
        df_en_fr["context"].tolist(),
        df_en_fr["corrupted_context"].tolist(),
        df_en_fr["context_answers"].tolist(),
        batch_size=64,
    )

    torch.save(h, "results/logprobs_diff.pt")


if __name__ == "__main__":
    main()
