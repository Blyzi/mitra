from utils.data import ICLDataset


def main():
    dataset = ICLDataset(
        [
            ("Hello", "Hi"),
            ("How are you?", "I'm fine"),
            ("What is your name?", "I am a bot."),
            ("Goodbye", "See you later!"),
            ("Thank you", "You're welcome!"),
            ("What's the weather like?", "It's sunny."),
            ("Do you like music?", "Yes, I love it."),
            ("Can you help me?", "Of course!"),
        ]
    )
    nshot_prompts, prompts, answers = dataset.get_prompts(
        1, "Q: {x}\nA: {y}\n", "You are a helpful assistant.\n"
    )
    for nshot_prompt, prompt, answer in zip(nshot_prompts, prompts, answers):
        print(f"N-shot Prompt:\n{nshot_prompt}\nPrompt:\n{prompt}\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
