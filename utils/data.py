import random


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
    ) -> tuple[list[str], list[str]]:
        nshot_prompts = []
        prompts = []
        answers = []

        print(len(self.x) // (n_shot + 1))

        for i in range(0, len(self.x) - n_shot, n_shot + 1):
            prompt = preprompt
            for j in range(n_shot):
                prompt += n_shot_format.format(x=self.x[i + j], y=self.y[i + j])
            prompt += question_format.format(x=self.x[i + n_shot], y="")

            nshot_prompts.append(prompt)
            prompts.append(preprompt + question_format.format(x=self.x[i + n_shot]))
            answers.append(self.y[i + n_shot])
        return nshot_prompts, prompts, answers

    def get_token_indexes(self, prompts: list[str], token: str, tokenizer) -> list[int]:
        token_indexes = []
        for prompt in prompts:
            tokens = tokenizer.tokenize(prompt)
            last_index = len(tokens) - 1 - tokens[::-1].index(token)
            token_indexes.append(last_index)
        return token_indexes

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.x[idx], self.y[idx]
