from src.utils.models import Model


def get_gemma_model_size(name: str) -> int:
    if "gemma-3-270m" in name:
        return 0.27
    elif "gemma-3-1b" in name:
        return 1
    elif "gemma-3-4b" in name:
        return 4
    elif "gemma-3-12b" in name:
        return 12
    elif "gemma-3-27b" in name:
        return 27
    else:
        raise ValueError(f"Unknown Gemma-3 model size in name: {name}")


class Gemma3Model(Model):
    def __init__(self, name):
        if get_gemma_model_size(name) >= 4:
            super().__init__(
                name=name,
                get_layers_func=lambda llm: llm.model.language_model.layers,
                get_out_proj_func=lambda self_attn: self_attn.o_proj,
                get_self_attn_func=lambda layer: layer.self_attn,
                get_n_head_func=lambda llm: llm.config.text_config.num_attention_heads,
                get_d_head_func=lambda llm: llm.config.text_config.head_dim,
                get_d_llm_func=lambda llm: llm.config.text_config.hidden_size,
            )
        else:
            super().__init__(
                name=name,
                get_layers_func=lambda llm: llm.model.layers,
                get_out_proj_func=lambda self_attn: self_attn.o_proj,
                get_self_attn_func=lambda layer: layer.self_attn,
                get_n_head_func=lambda llm: llm.config.num_attention_heads,
                get_d_head_func=lambda llm: llm.config.head_dim,
                get_d_llm_func=lambda llm: llm.config.hidden_size,
            )
