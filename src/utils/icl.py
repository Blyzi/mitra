import random
from transformers import PreTrainedTokenizer
import pandas as pd

from src.utils.functions import random_derangement


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
        self,
        n_shot: int,
        n_shot_format: str,
        question_format: str,
        preprompt: str = "",
        local_corruption: bool = False,
    ) -> pd.DataFrame:
        context = []
        corrupted_context = []
        context_answers = []

        for i in range(0, len(self.x) - n_shot - 1, n_shot + 1):
            if local_corruption:
                corrupted_choices = random_derangement(self.y[i : i + n_shot])
            else:
                corrupted_choices = random.sample(
                    self.y[0:i] + self.y[i + n_shot + 1 :], k=n_shot
                )  # Sample from all answers except the current context

            context.append(
                preprompt
                + "".join(
                    [
                        n_shot_format.format(x=self.x[i + j], y=self.y[i + j])
                        for j in range(n_shot)
                    ]
                )
                + question_format.format(x=self.x[i + n_shot])
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
                + question_format.format(x=self.x[i + n_shot])
            )
            context_answers.append(self.y[i + n_shot])

        return pd.DataFrame(
            {
                "context": context,
                "corrupted_context": corrupted_context,
                "context_answers": context_answers,
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
