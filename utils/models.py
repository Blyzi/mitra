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

    def get_fv(
        self,
        prompts: list[str],
        corrupted_prompts: list[str],
        answers: list[str],
        prompts_tokens_idx: list[int],
        corrupted_prompts_tokens_idx: list[int],
        batch_size: int = 1,
    ) -> torch.Tensor:
        heads = range(self.llm.config.num_attention_heads)
        n_samples = len(prompts)
        correct_completion_ids = [
            toks[0] for toks in self.llm.tokenizer(answers)["input_ids"]
        ]

        with torch.no_grad():
            z_dict = {}
            head_tensor = torch.empty(
                (n_samples, len(self.layers), self.n_head, self.d_head),
                device=self.llm.device,
            )

            for i, prompt in enumerate(tqdm(prompts, desc="Regular Prompts")):
                with self.llm.trace(prompt):
                    for layer in range(len(self.layers)):
                        z = getattr(
                            getattr(self.layers[layer], self.attn_key),
                            self.out_proj_key,
                        ).input[:, prompts_tokens_idx[i]]

                        z_reshaped = z.reshape(1, self.n_head, self.d_head)

                        for head in heads:
                            head_tensor[i, layer, head] = z_reshaped[0, head].save()

            for layer in range(len(self.layers)):
                for head in heads:
                    z_dict[(layer, head)] = head_tensor[:, layer, head].mean(dim=0)

            correct_logprobs_corrupted = torch.empty(
                (n_samples,), device=self.llm.device
            )
            for i, corrupted_prompt in enumerate(
                tqdm(corrupted_prompts, desc="Corrupted prompts")
            ):
                with self.llm.trace(corrupted_prompt) as tracer:
                    correct_logprobs_corrupted[i] = (
                        self.llm.lm_head.output[:, -1]
                        .log_softmax(dim=-1)[0, correct_completion_ids[i]]
                        .save()
                    )

            correct_logprobs_dict = {}
            for layer in tqdm(range(12, 14), desc="Layers", position=0):
                for head in tqdm(heads, desc="Heads", position=1):
                    correct_logprobs = torch.empty((n_samples,), device=self.llm.device)

                    for i, batch_corrupted_prompts, current_batch_size in batch(
                        corrupted_prompts, batch_size
                    ):
                        with self.llm.trace(batch_corrupted_prompts) as tracer:
                            # Get hidden states, reshape to get head dimension, then set it to the a-vector
                            z = getattr(
                                getattr(self.layers[layer], self.attn_key),
                                self.out_proj_key,
                            ).input[
                                torch.arange(current_batch_size),
                                corrupted_prompts_tokens_idx[
                                    i : i + current_batch_size
                                ],
                            ]

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
                # layers=len(self.layers),
                layers=2,
            )

            print(
                "torch.stack(list(correct_logprobs_dict.values()))",
                torch.stack(list(correct_logprobs_dict.values())).shape,
            )

            print(
                "all_correct_logprobs_intervention",
                all_correct_logprobs_intervention.shape,
            )
            print("correct_logprobs_corrupted", correct_logprobs_corrupted.shape)

            print(all_correct_logprobs_intervention)
            print(correct_logprobs_corrupted)

            logprobs_diff = (
                all_correct_logprobs_intervention - correct_logprobs_corrupted
            )  # shape [layers heads n_samples]

            print("logprobs_diff", logprobs_diff.shape)

            # Return mean effect of intervention, over the batch dimension
            return logprobs_diff.mean(dim=-1)

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
