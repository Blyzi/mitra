from collections import defaultdict
from typing import Any, Callable
import einops
from nnsight import LanguageModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.functions import batch
import gc
import random


class Model:
    def __init__(
        self,
        name,
        get_layers_func: Callable[[LanguageModel], list],
        get_self_attn_func: Callable[[Any], Any],
        get_out_proj_func: Callable[[Any], Any],
        get_n_head_func: Callable[[LanguageModel], Any],
        get_d_head_func: Callable[[LanguageModel], Any],
        get_d_llm_func: Callable[[LanguageModel], Any],
    ):
        self.name = name
        self.llm = LanguageModel(
            name, device_map="auto", torch_dtype=torch.bfloat16, dispatch=True
        )

        self.tokenizer = self.llm.tokenizer
        self.space_token_id = self.tokenizer.encode(" ")[0]

        self.get_self_attn_func = get_self_attn_func
        self.get_out_proj_func = get_out_proj_func
        self.get_layers_func = get_layers_func

        self.layers = get_layers_func(self.llm)
        self.n_head = get_n_head_func(self.llm)
        self.d_head = get_d_head_func(self.llm)
        self.d_llm = get_d_llm_func(self.llm)
        self.n_layers = len(get_layers_func(self.llm))

    def get_representations(
        self, prompts: list[str], tokens_idx: list[int]
    ) -> torch.Tensor:
        h = torch.empty(
            (len(prompts), self.n_layers, self.d_llm),
            device=self.llm.device,
        )

        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts)):
                with self.llm.trace(prompt):
                    for layer in range(self.n_layers):
                        h[i, layer, :] = self.get_output(self.get_layers()[layer])[
                            tokens_idx[i], :
                        ]

        return h.mean(dim=0)

    def get_kl_divergence(
        self,
        prompts_p: list[str],
        prompts_q: list[str],
        windows: list[tuple[int, int]],
        batch_size: int = 1,
    ) -> list[torch.Tensor]:
        kl_divergences = []
        prompts_pairs = list(zip(prompts_p, prompts_q))

        with torch.no_grad():
            for i, batch_pairs, current_batch_size in tqdm(
                batch(prompts_pairs, batch_size=batch_size),
                desc="KL Divergence Batches",
            ):
                with self.llm.trace([pair[0] for pair in batch_pairs]):
                    p_logprobs = self.llm.lm_head.output.log_softmax(dim=-1).save()

                with self.llm.trace([pair[1] for pair in batch_pairs]):
                    q_logprobs = self.llm.lm_head.output.log_softmax(dim=-1).save()

                for j in range(current_batch_size):
                    kl_divergence = F.kl_div(
                        q_logprobs[j][windows[i + j][0] : windows[i + j][1]],
                        p_logprobs[j][windows[i + j][0] : windows[i + j][1]],
                        log_target=True,
                        reduction="none",
                    ).sum(dim=-1)

                    kl_divergences.append(kl_divergence)

                del p_logprobs
                del q_logprobs

        return kl_divergences

    def get_activation_patch_map(
        self,
        prompts: list[str],
        corrupted_prompts: list[str],
        answers: list[str],
        batch_size: int = 1,
        selected_heads: list[tuple[int, int]] = None,
    ) -> torch.Tensor:
        heads = range(self.n_head)
        n_samples = len(prompts)
        answer_tokens = self.tokenizer(
            answers,
            add_special_tokens=False,
        )["input_ids"]

        full_prompts = [prompts[i] + answers[i] for i in range(len(answers))]
        full_corrupted_prompts = [
            corrupted_prompts[i] + answers[i] for i in range(len(answers))
        ]

        kl_divergences = self.get_kl_divergence(
            full_prompts,
            full_corrupted_prompts,
            # We start from -len(answer_tokens[i]) - 1 to -1 to only consider answer tokens starting from the last token of the question
            windows=[(-len(answer_tokens[i]) - 1, -1) for i in range(n_samples)],
            batch_size=batch_size,
        )

        max_kl_answer_indices = [
            torch.argmax(kl_divergences[i]).item() for i in range(n_samples)
        ]

        # Get the token IDs of the correct completions after the max KL divergence token
        correct_completion_ids = [
            # We don't add 1 because there is a shift of -1 because we start from the last token of the question
            answer_tokens[i][max_kl_answer_indices[i]]
            for i in range(n_samples)
        ]

        # Build the prompts up to the max KL divergence token
        patch_prompts = [
            prompts[i]
            + self.tokenizer.decode(answer_tokens[i][: max_kl_answer_indices[i]])
            for i in range(n_samples)
        ]
        patch_corrupted_prompts = [
            corrupted_prompts[i]
            + self.tokenizer.decode(answer_tokens[i][: max_kl_answer_indices[i]])
            for i in range(n_samples)
        ]

        with torch.no_grad():
            z_dict = {}
            head_tensor = torch.zeros(
                (n_samples, self.n_layers, self.n_head, self.d_head),
                device=self.llm.device,
            )

            # We compute the mean activations for each head over the patch prompts
            for i, patch_prompt in enumerate(
                tqdm(patch_prompts, desc="Regular Prompts")
            ):
                with self.llm.trace(patch_prompt):
                    for layer in range(self.n_layers):
                        z = self.get_out_proj(
                            self.get_self_attn(self.layers[layer]),
                        ).input[0, -1]

                        z_reshaped = z.reshape(self.n_head, self.d_head)

                        for head in heads:
                            head_tensor[i, layer, head] = z_reshaped[head].save()

            for layer in range(len(self.layers)):
                for head in heads:
                    z_dict[(layer, head)] = head_tensor[:, layer, head].mean(dim=0)

            # We compute the correct logprobs for the corrupted prompts
            correct_logprobs_corrupted = torch.zeros(
                (n_samples,), device=self.llm.device
            )

            for i, patch_corrupted_prompt in enumerate(
                tqdm(patch_corrupted_prompts, desc="Corrupted prompts")
            ):
                with self.llm.trace(patch_corrupted_prompt):
                    logits = self.llm.lm_head.output[0, -1]

                    correct_logprobs_corrupted[i] = logits.log_softmax(dim=-1)[
                        correct_completion_ids[i]
                    ].save()

            # We now do the intervention by patching in the mean activations head by head
            correct_logprobs_intervention = torch.zeros(
                (self.n_layers, self.n_head, n_samples), device=self.llm.device
            )

            for layer in tqdm(range(len(self.layers)), desc="Layers", position=0):
                for head in tqdm(heads, desc="Heads", position=1):
                    if (
                        selected_heads is not None
                        and (layer, head) not in selected_heads
                    ):
                        continue

                    correct_logprobs = torch.zeros((n_samples,), device=self.llm.device)

                    for i, batch_patch_corrupted_prompts, current_batch_size in batch(
                        patch_corrupted_prompts, batch_size
                    ):
                        with self.llm.trace(batch_patch_corrupted_prompts):
                            # Get hidden states, reshape to get head dimension, then set it to the action vector
                            z = self.get_out_proj(
                                self.get_self_attn(self.layers[layer]),
                            ).input[:, -1]

                            z.reshape(current_batch_size, self.n_head, self.d_head)[
                                :, head
                            ] = z_dict[(layer, head)]

                            # Get logprobs at the end, which we'll compare with our corrupted logprobs
                            batch_correct_logprobs = (
                                self.llm.lm_head.output[:, -1]
                                .log_softmax(dim=-1)[
                                    torch.arange(current_batch_size),
                                    correct_completion_ids[i : i + current_batch_size],
                                ]
                                .save()
                            )

                        correct_logprobs[i : i + current_batch_size] = (
                            batch_correct_logprobs
                        )

                    correct_logprobs_intervention[layer, head] = correct_logprobs

                gc.collect()
                torch.cuda.empty_cache()

            logprobs_diff = torch.zeros(
                (self.n_layers, self.n_head, n_samples), device=self.llm.device
            )

            for layer in range(len(self.layers)):
                for head in heads:
                    if selected_heads is None or (layer, head) in selected_heads:
                        logprobs_diff[layer, head, :] = (
                            correct_logprobs_intervention[layer, head, :]
                            - correct_logprobs_corrupted
                        )

            return logprobs_diff

    def get_attribution_patch_map(
        self,
        clean_context: list[str],
        corrupted_context: list[str],
        answers: list[str],
        batch_size: int = 1,
    ) -> torch.Tensor:
        n_samples = len(clean_context)
        clean_out = []
        corrupted_out = []
        corrupted_grads = []

        answer_tokens = self.tokenizer(
            [f" {answers[i]}" for i in range(len(answers))],
            add_special_tokens=False,
        )["input_ids"]

        full_prompts = [f"{clean_context[i]} {answers[i]}" for i in range(len(answers))]
        full_corrupted_prompts = [
            f"{corrupted_context[i]} {answers[i]}" for i in range(len(answers))
        ]

        kl_divergences = self.get_kl_divergence(
            full_prompts,
            full_corrupted_prompts,
            # We start from -len(answer_tokens[i]) - 1 to -1 to only consider answer tokens starting from the last token of the question
            windows=[(-len(answer_tokens[i]) - 1, -1) for i in range(n_samples)],
            batch_size=batch_size,
        )

        max_kl_answer_indices = [
            torch.argmax(kl_divergences[i]).item() for i in range(n_samples)
        ]

        # Build the prompts up to the max KL divergence token
        patch_corrupted_prompts = [
            corrupted_context[i]
            + self.tokenizer.decode(answer_tokens[i][: max_kl_answer_indices[i]])
            for i in range(n_samples)
        ]
        patch_clean_prompts = [
            clean_context[i]
            + self.tokenizer.decode(answer_tokens[i][: max_kl_answer_indices[i]])
            for i in range(n_samples)
        ]

        for i, batch_patch_corrupted_prompt, current_batch_size in tqdm(
            batch(patch_corrupted_prompts, batch_size), desc="Corrupted Prompts"
        ):
            batch_corrupted_out = []
            batch_corrupted_grads = []

            with self.llm.trace(batch_patch_corrupted_prompt):
                for layer in range(len(self.layers)):
                    attn_out = self.get_out_proj(
                        self.get_self_attn(self.layers[layer]),
                    ).input.save()

                    attn_out.retain_grad()
                    batch_corrupted_out.append(attn_out)

                self.llm.lm_head.output[:, -1, :].sum().backward()

            for layer in range(len(self.layers)):
                batch_corrupted_grads.append(
                    batch_corrupted_out[layer].grad[:, -1].clone()
                )

                # Free up some memory
                with torch.no_grad():
                    original_batch_corrupted_out = batch_corrupted_out[layer]
                    clone_batch_corrupted_out = batch_corrupted_out[layer][
                        :, -1
                    ].clone()

                    batch_corrupted_out[layer] = clone_batch_corrupted_out
                    del original_batch_corrupted_out

                    torch.cuda.empty_cache()

            if len(corrupted_out) != 0:
                for layer in range(len(self.layers)):
                    corrupted_out[layer] = torch.cat(
                        (
                            corrupted_out[layer],
                            batch_corrupted_out[layer],
                        ),
                        dim=0,
                    )

                    corrupted_grads[layer] = torch.cat(
                        (
                            corrupted_grads[layer],
                            batch_corrupted_grads[layer],
                        ),
                        dim=0,
                    )
            else:
                corrupted_out = batch_corrupted_out
                corrupted_grads = batch_corrupted_grads

        with torch.no_grad():
            for i, batch_patch_clean_prompt, current_batch_size in tqdm(
                batch(patch_clean_prompts, batch_size), desc="Clean Prompts"
            ):
                batch_clean_out = []
                with self.llm.trace(batch_patch_clean_prompt):
                    for layer in range(len(self.layers)):
                        attn_out = self.get_out_proj(
                            self.get_self_attn(self.layers[layer]),
                        ).input
                        batch_clean_out.append(attn_out[:, -1].save())

                if len(clean_out) != 0:
                    for layer in range(len(self.layers)):
                        clean_out[layer] = torch.cat(
                            (
                                clean_out[layer],
                                batch_clean_out[layer],
                            ),
                            dim=0,
                        )
                else:
                    clean_out = batch_clean_out

            patching_results = torch.zeros(
                (self.n_layers, self.n_head),
                device=self.llm.device,
            )

            for corrupted_grad, corrupted, clean, layer in zip(
                corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
            ):
                residual_attr = einops.reduce(
                    corrupted_grad * (clean - corrupted),
                    "batch (head dim) -> head",
                    "mean",
                    head=self.n_head,
                    dim=self.d_head,
                )

                patching_results[layer] = residual_attr

            del corrupted_out
            del corrupted_grads
            del clean_out
            torch.cuda.empty_cache()

            return patching_results

    def calculate_fn_vector(
        self,
        clean_context: list[str],
        corrupted_context: list[str],
        answers: list[str],
        head_list: list[tuple[int, int]],
        batch_size: int = 1,
    ) -> dict[int, torch.Tensor]:
        """
        Calculates a function vector for the given heads, by averaging the output projection inputs
        over the given prompts.

        Inputs:
            model: LanguageModel
                the transformer you're doing this computation with
            dataset: ICLDataset
                the dataset of clean prompts from which we'll extract the function vector (we'll also
                create a corrupted version of this dataset for interventions)
            head_list: list[tuple[int, int]]
                list of attention heads we're calculating the function vector from
        """
        n_samples = len(clean_context)
        answer_tokens = self.tokenizer(
            [f" {answers[i]}" for i in range(len(answers))],
            add_special_tokens=False,
        )["input_ids"]

        full_prompts = [f"{clean_context[i]} {answers[i]}" for i in range(len(answers))]
        full_corrupted_prompts = [
            f"{corrupted_context[i]} {answers[i]}" for i in range(len(answers))
        ]

        kl_divergences = self.get_kl_divergence(
            full_prompts,
            full_corrupted_prompts,
            # We start from -len(answer_tokens[i]) - 1 to -1 to only consider answer tokens starting from the last token of the question
            windows=[(-len(answer_tokens[i]) - 1, -1) for i in range(n_samples)],
            batch_size=batch_size,
        )

        max_kl_answer_indices = [
            torch.argmax(kl_divergences[i]).item() for i in range(n_samples)
        ]

        # Build the prompts up to the max KL divergence token
        patch_clean_prompts = [
            clean_context[i]
            + self.tokenizer.decode(answer_tokens[i][: max_kl_answer_indices[i]])
            for i in range(n_samples)
        ]

        # Turn head_list into a dict of {layer: heads we need in this layer}
        head_dict = defaultdict(set)
        for layer, head in head_list:
            head_dict[layer].add(head)

        fn_vector_dict = {}

        with torch.no_grad():
            for layer, head_list in head_dict.items():
                hidden_states = torch.zeros(
                    (self.n_head * self.d_head),
                    device=self.llm.device,
                    dtype=torch.bfloat16,
                )

                for i, batch_prompts, current_batch_size in batch(
                    patch_clean_prompts, batch_size
                ):
                    batch_heads_norms = torch.zeros(
                        (self.n_head,), device=self.llm.device
                    )
                    with self.llm.trace(batch_prompts):
                        # Get the output projection layer
                        # out_proj = self.llm.transformer.h[layer].attn.out_proj
                        out_proj = self.get_out_proj(
                            self.get_self_attn(self.layers[layer]),
                        )

                        # Get the mean output projection input (note, setting values of this tensor will not
                        # have downstream effects on other tensors)
                        batch_hidden_states = out_proj.input[:, -1].mean(dim=0).save()

                        # Compute the mean norm for each head that is

                        for head in head_list:
                            batch_heads_norms[head] = (
                                out_proj.input[:, -1]
                                .reshape(current_batch_size, self.n_head, self.d_head)[
                                    :, head, :
                                ]
                                .norm(dim=-1)
                                .mean()
                                .save()
                            )

                    hidden_states += (
                        batch_hidden_states * current_batch_size / n_samples
                    )

                # Zero-ablate all heads which aren't in our list, then get the output (which
                # will be the sum over the heads we actually do want!)
                heads_to_ablate = set(range(self.n_head)) - head_dict[layer]
                print(
                    f"Layer {layer}, ablating heads: {heads_to_ablate}, keeping heads: {head_list}"
                )

                # Ablate the unimportant heads
                for head in heads_to_ablate:
                    hidden_states.reshape(self.n_head, self.d_head)[head] = 0.0

                with self.llm.trace(" ") as tracer:
                    out_proj = self.get_out_proj(
                        self.get_self_attn(self.layers[layer]),
                    )

                    # Now that we've zeroed all unimportant heads, get the output & add it to the list
                    # (we need a single batch dimension so we can use `out_proj`)
                    out_proj_output = out_proj(hidden_states.unsqueeze(0)).squeeze()

                    fn_vector_dict[layer] = out_proj_output.save()

        return fn_vector_dict

    def calculate_head_output(
        self,
        clean_context: list[str],
        corrupted_context: list[str],
        answers: list[str],
        head: tuple[int, int],
        batch_size: int = 1,
    ) -> torch.Tensor:
        n_samples = len(clean_context)
        answer_tokens = self.tokenizer(
            [f" {answers[i]}" for i in range(len(answers))],
            add_special_tokens=False,
        )["input_ids"]

        full_prompts = [f"{clean_context[i]} {answers[i]}" for i in range(len(answers))]
        full_corrupted_prompts = [
            f"{corrupted_context[i]} {answers[i]}" for i in range(len(answers))
        ]

        kl_divergences = self.get_kl_divergence(
            full_prompts,
            full_corrupted_prompts,
            # We start from -len(answer_tokens[i]) - 1 to -1 to only consider answer tokens starting from the last token of the question
            windows=[(-len(answer_tokens[i]) - 1, -1) for i in range(n_samples)],
            batch_size=batch_size,
        )

        max_kl_answer_indices = [
            torch.argmax(kl_divergences[i]).item() for i in range(n_samples)
        ]

        # Build the prompts up to the max KL divergence token
        patch_clean_prompts = [
            clean_context[i]
            + self.tokenizer.decode(answer_tokens[i][: max_kl_answer_indices[i]])
            for i in range(n_samples)
        ]

        head_output = torch.zeros((self.d_head,), device=self.llm.device)

        with torch.no_grad():
            for i, batch_prompts, current_batch_size in batch(
                patch_clean_prompts, batch_size
            ):
                with self.llm.trace(batch_prompts):
                    # Get the output projection layer
                    # out_proj = self.llm.transformer.h[layer].attn.out_proj
                    out_proj = self.get_out_proj(
                        self.get_self_attn(self.layers[head[0]]),
                    )

                    # Get the mean output projection input (note, setting values of this tensor will not
                    # have downstream effects on other tensors)
                    batch_hidden_states = out_proj.input[:, -1].mean(dim=0).save()

                # hidden_states += batch_hidden_states * current_batch_size / n_samples
                head_output += (
                    batch_hidden_states.reshape(self.n_head, self.d_head)[head[1]]
                    * current_batch_size
                    / n_samples
                )

        return head_output

    def generate_with_fn_vector(
        self,
        prompts: list[str],
        fn_vector: torch.Tensor,
        max_new_tokens: int = 5,
        stops: list[str] = [],
    ) -> list[str]:
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

        fn_vector_device = {
            layer: fn_vector[layer].to(self.llm.device) for layer in fn_vector.keys()
        }

        stops_tokens = [
            self.tokenizer.encode(stop, add_special_tokens=False)[0] for stop in stops
        ] + [self.tokenizer.eos_token_id]

        for i, prompt in (
            pbar := tqdm(
                enumerate(prompts),
                desc="Generating with function vector",
                total=len(prompts),
            )
        ):
            prompt_len = len(self.llm.tokenizer.encode(prompt))

            with torch.no_grad():
                with self.llm.generate(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=stops_tokens,
                ) as tracer:
                    with tracer.invoke(prompt):
                        if len(fn_vector_device.keys()) > 0:
                            with tracer.all():
                                for layer in fn_vector_device.keys():
                                    self.get_out_proj(
                                        self.get_self_attn(self.layers[layer])
                                    ).output[:, -1] += fn_vector_device[layer]

                    with tracer.invoke():
                        tokens_intervention = self.llm.generator.output[
                            0, prompt_len:-1
                        ].save()

            generation = self.tokenizer.decode(tokens_intervention)
            completion_intervention.append(generation.strip().split("\n")[0])

            if i % 10 == 0 and i > 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Monitor GPU memory usage
            if torch.cuda.is_available():
                used = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                pbar.set_postfix(cuda_mem=f"{100 * used / total:.1f}%")

        return completion_intervention

    def generate_with_ablation(
        self,
        prompts: list[str],
        heads_to_ablate: list[tuple[int, int]],
        max_new_tokens: int = 5,
        stops: list[str] = [],
        random_ablation: bool = False,
    ) -> list[str]:
        """
        Generates completions while ablating specific attention heads at a given layer.

        Inputs:
            prompts: list[str]
                The list of prompts to generate completions for.
            heads_to_ablate: list[tuple[int, int]]
                The list of attention head indices to ablate.
            max_new_tokens: int
                The number of additional tokens to generate.
            stops: list[str]
                List of stop tokens to end generation.
            random_ablation: bool
                If True, randomly selects heads to ablate instead of using provided list.

        Returns:
            completions: list[str]
                The list of generated completions with ablation.
        """

        completions = []

        stops_tokens = [
            self.tokenizer.encode(stop, add_special_tokens=False)[0] for stop in stops
        ] + [self.tokenizer.eos_token_id]

        # Group heads to ablate by layer
        ablation_dict = defaultdict(list)
        for layer, head in heads_to_ablate:
            ablation_dict[layer].append(head)

        for i, prompt in (
            pbar := tqdm(
                enumerate(prompts),
                desc="Generating with ablation",
                total=len(prompts),
            )
        ):
            prompt_len = len(self.llm.tokenizer.encode(prompt))

            # If random ablation is enabled, replace heads_to_ablate with random heads at random layers
            if random_ablation:
                ablation_dict = defaultdict(list)
                for _ in range(len(heads_to_ablate)):
                    random_layer = random.randint(0, self.n_layers - 1)
                    random_head = random.randint(0, self.n_head - 1)
                    ablation_dict[random_layer].append(random_head)

            with torch.no_grad():
                with self.llm.generate(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=stops_tokens,
                ) as tracer:
                    with tracer.invoke(prompt):
                        if len(heads_to_ablate) > 0:
                            with tracer.all():
                                for layer in sorted(ablation_dict.keys()):
                                    out_proj = self.get_out_proj(
                                        self.get_self_attn(self.layers[layer])
                                    )
                                    for head in ablation_dict[layer]:
                                        out_proj.input[:, -1].reshape(
                                            self.n_head, self.d_head
                                        )[head] = 0.0

                    with tracer.invoke():
                        tokens_ablation = self.llm.generator.output[
                            0, prompt_len:-1
                        ].save()

            generation = self.tokenizer.decode(tokens_ablation)
            completions.append(generation.strip().split("\n")[0])

            if i % 10 == 0 and i > 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Monitor GPU memory usage
            if torch.cuda.is_available():
                used = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                pbar.set_postfix(cuda_mem=f"{100 * used / total:.1f}%")

        return completions

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 5,
        stops: list[str] = [],
    ) -> list[str]:
        completion_baseline = []

        for i, prompt in tqdm(
            enumerate(prompts), desc="Generating baseline", total=len(prompts)
        ):
            prompt_len = len(self.llm.tokenizer.encode(prompt))

            with self.llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                ]
                + [
                    self.tokenizer.encode(stop, add_special_tokens=False)[0]
                    for stop in stops
                ],
            ):
                tokens_baseline = self.llm.generator.output[:, prompt_len:-1].save()

            tokens = self.tokenizer.decode(tokens_baseline[0])

            completion_baseline.append(tokens.strip().split("\n")[0])

            if i % 10 == 0 and i > 0:
                gc.collect()
                torch.cuda.empty_cache()

        return completion_baseline

    def get_self_attn(self, layer) -> Any:
        return self.get_self_attn_func(layer)

    def get_out_proj(self, self_attn) -> Any:
        return self.get_out_proj_func(self_attn)

    def __repr__(self):
        return self.llm.__repr__()
