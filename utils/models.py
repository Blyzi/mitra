from collections import defaultdict
from xml.parsers.expat import model
import einops
from nnsight import LanguageModel
import torch
from tqdm import tqdm
import re
import pandas as pd
from utils.functions import batch


class Model:
    def __init__(
        self,
        name,
        layers_adr: list[str],
        attn_key: str,
        out_proj_key: str,
        n_head_key: str,
        d_head_key: str,
    ):
        self.name = name
        self.llm = LanguageModel(
            name, device_map="auto", dtype=torch.bfloat16, dispatch=True
        )
        self.layers = self.llm
        self.tokenizer = self.llm.tokenizer
        self.attn_key = attn_key
        self.out_proj_key = out_proj_key

        for adr in layers_adr:
            self.layers = getattr(self.layers, adr)

        self.n_head = self.llm.config.__dict__[n_head_key]
        self.d_head = self.llm.config.__dict__[d_head_key]
        self.d_model = self.llm.config.hidden_size

    def get_representations(
        self, prompts: list[str], tokens_idx: list[int]
    ) -> torch.Tensor:
        h = torch.empty(
            (len(prompts), len(self.layers), self.llm.model.config.hidden_size),
            device=self.llm.device,
        )

        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts)):
                with self.llm.trace(prompt):
                    for layer in range(len(self.layers)):
                        h[i, layer, :] = self.layers[layer].output[0][tokens_idx[i], :]

        return h.mean(dim=0)

    def get_fv_impact(
        self,
        prompts: list[str],
        corrupted_prompts: list[str],
        answers: list[str],
        batch_size: int = 1,
    ) -> torch.Tensor:
        heads = range(self.llm.config.num_attention_heads)
        n_samples = len(prompts)
        correct_completion_ids = [
            toks[0]
            for toks in self.tokenizer(
                answers,
                add_special_tokens=False,
            )["input_ids"]
        ]

        with torch.no_grad():
            z_dict = {}
            head_tensor = torch.zeros(
                (n_samples, len(self.layers), self.n_head, self.d_head),
                device=self.llm.device,
            )

            for i, prompt in enumerate(tqdm(prompts, desc="Regular Prompts")):
                with self.llm.trace(prompt):
                    for layer in range(len(self.layers)):
                        z = getattr(
                            getattr(self.layers[layer], self.attn_key),
                            self.out_proj_key,
                        ).input[:, -1]

                        z_reshaped = z.reshape(1, self.n_head, self.d_head)

                        for head in heads:
                            head_tensor[i, layer, head] = z_reshaped[0, head].save()

            for layer in range(len(self.layers)):
                for head in heads:
                    z_dict[(layer, head)] = head_tensor[:, layer, head].mean(dim=0)

            correct_logprobs_corrupted = torch.zeros(
                (n_samples,), device=self.llm.device
            )
            for i, corrupted_prompt in enumerate(
                tqdm(corrupted_prompts, desc="Corrupted prompts")
            ):
                with self.llm.trace(corrupted_prompt) as tracer:
                    logits = self.llm.lm_head.output[:, -1]

                    correct_logprobs_corrupted[i] = logits.log_softmax(dim=-1)[
                        :, correct_completion_ids[i]
                    ].save()

            correct_logprobs_dict = {}
            for layer in tqdm(range(len(self.layers)), desc="Layers", position=0):
                for head in tqdm(heads, desc="Heads", position=1):
                    correct_logprobs = torch.zeros((n_samples,), device=self.llm.device)

                    for i, batch_corrupted_prompts, current_batch_size in batch(
                        corrupted_prompts, batch_size
                    ):
                        with self.llm.trace(batch_corrupted_prompts) as tracer:
                            # Get hidden states, reshape to get head dimension, then set it to the a-vector
                            z = getattr(
                                getattr(self.layers[layer], self.attn_key),
                                self.out_proj_key,
                            ).input[:, -1]
                            # Can be replace by:
                            # torch.arange(current_batch_size),
                            # corrupted_prompts_tokens_idx[
                            #     i : i + current_batch_size
                            # ],

                            z.reshape(current_batch_size, self.n_head, self.d_head)[
                                :, head
                            ] = z_dict[(layer, head)]

                            # Get logprobs at the end, which we'll compare with our corrupted logprobs
                            correct_logprobs[i : i + current_batch_size] = (
                                self.llm.lm_head.output[:, -1]
                                .log_softmax(dim=-1)[
                                    torch.arange(current_batch_size),
                                    correct_completion_ids[i : i + current_batch_size],
                                ]
                                .save()
                            )

                    correct_logprobs_dict[(layer, head)] = correct_logprobs

            # Get difference between intervention logprobs and corrupted logprobs, and take mean over batch dim
            all_correct_logprobs_intervention = einops.rearrange(
                torch.stack(list(correct_logprobs_dict.values())),
                "(layers heads) batch -> layers heads batch",
                layers=len(self.layers),
                # layers=4,
            )

            logprobs_diff = (
                all_correct_logprobs_intervention - correct_logprobs_corrupted
            )  # shape [layers heads n_samples]

            # Return mean effect of intervention, over the batch dimension
            return logprobs_diff

    def calculate_fn_vector(
        self,
        prompts: list[str],
        head_list: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Returns a vector of length `d_model`, containing the sum of vectors written to the residual
        stream by the attention heads in `head_list`, averaged over all inputs in `dataset`.

        Inputs:
            model: LanguageModel
                the transformer you're doing this computation with
            dataset: ICLDataset
                the dataset of clean prompts from which we'll extract the function vector (we'll also
                create a corrupted version of this dataset for interventions)
            head_list: list[tuple[int, int]]
                list of attention heads we're calculating the function vector from
        """
        # Turn head_list into a dict of {layer: heads we need in this layer}
        head_dict = defaultdict(set)
        for layer, head in head_list:
            head_dict[layer].add(head)

        fn_vector_dict = {}
        with torch.no_grad():
            with self.llm.trace(prompts):
                for layer, head_list in head_dict.items():
                    # Get the output projection layer
                    # out_proj = self.llm.transformer.h[layer].attn.out_proj
                    out_proj = getattr(
                        getattr(self.layers[layer], self.attn_key), self.out_proj_key
                    )

                    # Get the mean output projection input (note, setting values of this tensor will not
                    # have downstream effects on other tensors)
                    hidden_states = out_proj.input[:, -1].mean(dim=0)

                    # Zero-ablate all heads which aren't in our list, then get the output (which
                    # will be the sum over the heads we actually do want!)
                    heads_to_ablate = set(range(self.n_head)) - head_dict[layer]
                    print(
                        f"Layer {layer}, ablating heads: {heads_to_ablate}, keeping heads: {head_list}"
                    )

                    for head in heads_to_ablate:
                        hidden_states.reshape(self.n_head, self.d_head)[head] = 0.0

                    print(f"Hidden states max: {hidden_states.max()}")

                    # Now that we've zeroed all unimportant heads, get the output & add it to the list
                    # (we need a single batch dimension so we can use `out_proj`)
                    out_proj_output = out_proj(hidden_states.unsqueeze(0)).squeeze()

                    fn_vector_dict[layer] = out_proj_output

        return fn_vector_dict

    def generate_with_fn_vector(
        self,
        prompts: list[str],
        fn_vector: torch.Tensor,
        max_new_tokens: int = 5,
        stops: list[str] = [],
    ) -> tuple[str, str]:
        """
        Intervenes with a function vector, by adding it at the last sequence position of a generated
        prompt.

        Inputs:
            model: LanguageModel
                the transformer you're doing this computation with
            word: str
                The word substituted into the prompt template, via prompt_template.format(x=word)
            layer: int
                The layer we'll make the intervention (by adding the function vector)
            fn_vector: Float[Tensor, "d_model"]
                The vector we'll add to the final sequence position for each new token to be generated
            prompt_template:
                The template of the prompt we'll use to produce completions
            n_tokens: int
                The number of additional tokens we'll generate for our unsteered / steered completions

        Returns:
            completion: str
                The full completion (including original prompt) for the no-intervention case
            completion_intervention: str
                The full completion (including original prompt) for the intervention case
        """
        completion_intervention = []

        pattern = r"\s*(?:" + "|".join(map(re.escape, stops)) + r")\s*"
        for prompt in tqdm(prompts, desc="Generating with function vector"):
            prompt_len = len(self.llm.tokenizer.encode(prompt))

            with self.llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            ) as generator:
                for layer in fn_vector.keys():
                    self.layers[layer].output[:, -1] += fn_vector[layer] * 5

                tokens_intervention = self.llm.generator.output[:, prompt_len:].save()

            tokens = self.tokenizer.decode(tokens_intervention[0])

            if stops:
                completion_intervention.append(re.split(pattern, tokens)[0])
            else:
                completion_intervention.append(tokens)

        return completion_intervention

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 2,
        stops: list[str] = [],
    ) -> list[str]:
        generated = []
        pattern = r"\s*(?:" + "|".join(map(re.escape, stops)) + r")\s*"

        for i, prompt in enumerate(tqdm(prompts)):
            prompt_len = len(self.llm.tokenizer.encode(prompt))

            with self.llm.generate(
                prompt, max_new_tokens=max_new_tokens, do_sample=False
            ) as generator:
                output = self.llm.generator.output[:, prompt_len:].save()

            tokens = self.llm.tokenizer.decode(output[0])

            if stops:
                generated.append(re.split(pattern, tokens)[0])
            else:
                generated.append(tokens)

        return pd.DataFrame(
            {
                "completion": generated,
            }
        )

    def generate_with_intervention(
        self,
        prompts: list[str],
        representation: torch.Tensor,
        layer: int,
        tokens_idx: list[int],
        max_new_tokens: int = 2,
        stops: list[str] = [],
    ) -> list[str]:
        generated_intervention = []
        pattern = r"\s*(?:" + "|".join(map(re.escape, stops)) + r")\s*"

        for i, prompt in enumerate(tqdm(prompts)):
            prompt_len = len(self.llm.tokenizer.encode(prompt))

            with self.llm.generate(
                prompt, max_new_tokens=max_new_tokens, do_sample=False
            ) as generator:
                for layer_idx in range(10, 14):
                    hidden_states = self.layers[layer_idx].output[0]

                    hidden_states[tokens_idx[i], :] += representation[layer_idx, :] * 8

                output_intervention = self.llm.generator.output[:, prompt_len:].save()

            tokens_intervention = self.llm.tokenizer.decode(output_intervention[0])

            if stops:
                generated_intervention.append(re.split(pattern, tokens_intervention)[0])
            else:
                generated_intervention.append(tokens_intervention)

        return pd.DataFrame(
            {
                "completion_intervention": generated_intervention,
            }
        )

    def compute_logprobs(
        self,
        prompts: list[str],
        answers: list[str],
        representation: torch.Tensor,
        layer: int,
        tokens_idx: list[int],
    ) -> list[float]:
        logprobs = []

        for prompt in tqdm(prompts):
            with self.llm.trace(prompt):
                logprob = (
                    self.llm.lm_head.logits[:, :-1]
                    .log_softmax(dim=-1)
                    .gather(2, self.llm.input_ids[:, 1:, None])
                )
                logprobs.append(logprob.sum().item())

        return logprobs
