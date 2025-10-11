from xml.parsers.expat import model
from nnsight import LanguageModel
import torch
from tqdm import tqdm
import re


class Model:
    def __init__(self, name, layers_adr: list[str]):
        self.name = name
        self.llm = LanguageModel(
            name, device_map="auto", dtype=torch.bfloat16, dispatch=True
        )
        self.layers = self.llm
        self.tokenizer = self.llm.tokenizer

        for adr in layers_adr:
            self.layers = getattr(self.layers, adr)

    def generate(
        self,
        nshot_prompts: list[str],
        prompts: list[str],
        layer: int,
        max_new_tokens: int = 5,
        stops: list[str] = [],
    ) -> list[str]:
        # generated = []
        # pattern = r"\s*(?:" + "|".join(map(re.escape, stops)) + r")\s*"

        # for prompt in tqdm(prompts):
        #     prompt_len = len(self.llm.tokenizer.encode(prompt))
        #     with self.llm.generate(
        #         prompt, max_new_tokens=max_new_tokens, do_sample=False
        #     ) as generator:
        #         # Save only the newly generated tokens
        #         output = self.llm.generator.output[:, prompt_len:].save()

        #     tokens = self.tokenizer.decode(output[0])

        #     if stops:
        #         generated.append(re.split(pattern, tokens)[0])
        #     else:
        #         generated.append(tokens)

        # return generated

        with self.llm.trace() as tracer:
            with tracer.invoke(nshot_prompts):
                print(self.layers[layer].output[0][:, -1].shape)
                h = self.layers[layer].output[0][:, -1].mean(dim=0)
                print(h)

            with tracer.invoke(prompts):
                clean_tokens = self.llm.lm_head.output[:, -1].argmax(dim=-1).save()

            with tracer.invoke(prompts):
                hidden = self.layers[layer].output[0]
                hidden[:, -1] += h
                intervene_tokens = self.llm.lm_head.output[:, -1].argmax(dim=-1).save()

        completions_zero_shot = self.tokenizer.batch_decode(clean_tokens)
        completions_intervention = self.tokenizer.batch_decode(intervene_tokens)
        return completions_zero_shot, completions_intervention

    def generate_with_intervention(
        self,
        prompts: list[str],
        representation: torch.Tensor,
        layer: int,
        tokens_idx: list[int],
        max_new_tokens: int = 5,
        stops: list[str] = [],
    ) -> list[str]:
        generated = []
        pattern = r"\s*(?:" + "|".join(map(re.escape, stops)) + r")\s*"

        for i, prompt in enumerate(prompts):
            prompt_len = len(self.llm.tokenizer.encode(prompt))

            # with self.llm.generate(
            #     prompt, max_new_tokens=max_new_tokens, do_sample=False
            # ) as generator:
            #     hidden_states = self.layers[layer].output[0][tokens_idx[i], :]

            #     hidden_states += representation[layer, :]

            #     output = self.llm.generator.output[:, prompt_len:].save()

            with self.llm.trace(prompt) as tracer:
                hidden_states = self.layers[layer].output[0]

                print(hidden_states.shape)

                hidden_states[-1, :] += representation[layer, :]

                output = self.llm.lm_head.output[:, -1:].argmax(dim=-1).save()

            tokens = self.llm.tokenizer.decode(output[0])

            if stops:
                generated.append(re.split(pattern, tokens)[0])
            else:
                generated.append(tokens)

        return generated
