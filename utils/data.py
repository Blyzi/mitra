import random
from transformers import PreTrainedTokenizer
import pandas as pd


class ICLDataset:
    def __init__(
        self, pairs: list[tuple[str, str]], bidirectional: bool = False, seed: int = 42
    ):
        random.seed(seed)
        random.shuffle(pairs)

        if bidirectional:
            pairs = [(a, b) if random.random() < 0.5 else (b, a) for a, b in pairs]

        self.x, self.y = zip(*pairs)

    def get_prompts(
        self, n_shot: int, n_shot_format: str, question_format: str, preprompt: str = ""
    ) -> pd.DataFrame:
        nshot_prompts = []
        context = []
        corrupted_context = []
        prompts = []
        corrupted_nshot_prompts = []
        answers = []

        for i in range(0, len(self.x) - n_shot, n_shot + 1):
            corrupted_choices = random.sample(self.y, k=n_shot)

            context.append(
                preprompt
                + "".join(
                    [
                        n_shot_format.format(x=self.x[i + j], y=self.y[i + j])
                        for j in range(n_shot)
                    ]
                )
            )
            corrupted_context.append(
                preprompt
                + "".join(
                    [
                        n_shot_format.format(
                            x=self.x[i + j],
                            y=corrupted_choices[j],
                        )
                        for j in range(n_shot)
                    ]
                )
            )

            nshot_prompts.append(
                preprompt
                + "".join(
                    [
                        n_shot_format.format(x=self.x[i + j], y=self.y[i + j])
                        for j in range(n_shot)
                    ]
                )
                + question_format.format(x=self.x[i + n_shot], y="")
            )
            corrupted_nshot_prompts.append(
                preprompt
                + "".join(
                    [
                        n_shot_format.format(
                            x=self.x[i + j],
                            y=corrupted_choices[j],
                        )
                        for j in range(n_shot)
                    ]
                )
                + question_format.format(x=self.x[i + n_shot], y="")
            )

            prompts.append(preprompt + question_format.format(x=self.x[i + n_shot]))
            answers.append(self.y[i + n_shot])

        return pd.DataFrame(
            {
                "nshot_prompt": nshot_prompts,
                "corrupted_nshot_prompt": corrupted_nshot_prompts,
                "noshot_prompt": prompts,
                "context": context,
                "corrupted_context": corrupted_context,
                "answer": answers,
            }
        )

    def get_token_indexes(
        self, prompts: list[str], token: str, tokenizer: PreTrainedTokenizer
    ) -> pd.Series:
        token_indexes = []
        for prompt in prompts:
            tokens = tokenizer.tokenize(prompt)
            last_index = -1 - tokens[::-1].index(token)
            token_indexes.append(last_index)
        return pd.Series(token_indexes)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.x[idx], self.y[idx]
