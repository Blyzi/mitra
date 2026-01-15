from src.utils.models import Model


class Llama2Model(Model):
    def __init__(self, name):
        super().__init__(
            name=name,
            get_layers_func=lambda llm: llm.model.layers,
            get_out_proj_func=lambda self_attn: self_attn.o_proj,
            get_self_attn_func=lambda layer: layer.self_attn,
            get_n_head_func=lambda llm: llm.config.num_attention_heads,
            get_d_head_func=lambda llm: llm.config.head_dim,
            get_d_llm_func=lambda llm: llm.config.hidden_size,
        )
