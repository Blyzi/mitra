import torch
from utils.data import ICLDataset
from utils.models import Model
from datasets import load_dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    lang_en, lang_fr, lang_pt = "eng_Latn", "fra_Latn", "por_Latn"

    pairs_en_fr = load_dataset("facebook/flores", "all")["dev"].map(
        lambda x: {"pairs": (x["sentence_" + lang_en], x["sentence_" + lang_fr])}
    )["pairs"]

    icl_ds_en_fr = ICLDataset(pairs_en_fr, bidirectional=False)

    df_en_fr = icl_ds_en_fr.get_prompts(
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

    logprobs_diff = torch.load("results/logprobs_diff.pt").mean(dim=-1)

    selected_heads = [
        v for v in torch.nonzero(logprobs_diff > 0.1, as_tuple=False).tolist()
    ]

    print(f"Selected heads: {selected_heads}")

    h = model.calculate_fn_vector(df_en_fr["context"].tolist(), selected_heads)

    generations = model.generate_with_fn_vector(
        df_en_fr["noshot_prompt"].tolist(),
        fn_vector=h,
        max_new_tokens=5,
        stops=["\n"],
    )

    for i in range(30):
        print("Prompt:")
        print(df_en_fr["noshot_prompt"].tolist()[i])
        print("Generation:")
        print(generations[i])
        print("Ground Truth:")
        print(df_en_fr["noshot_answers"].tolist()[i])
        print("-----")


if __name__ == "__main__":
    main()
