# MITra

Code for the paper [_Translation Heads: Disentangling meaning from language in LLM-based machine translation_](https://arxiv.org/abs/2602.04613), published at ICML 2026.

## Overview

This repository investigates how large language models perform machine translation by identifying and analyzing two disentangled circuits in the attention heads:

- **Translation heads** — responsible for converting the *meaning* of a sentence across languages (identified by patching activations during source → target translation).
- **Language heads** — responsible for selecting the *output language* (identified by contrasting correct vs. random target language outputs).

We use activation patching to locate these heads, then steer and ablate them to understand their linguistic properties across language pairs and model families.

## Models & Data

**Models:** Llama 2 (7B), Llama 3.2 (1B, 3B), Gemma 3 (270M, 1B, 4B, 12B, 27B), Qwen 3 (0.6B, 1.7B, 4B)

**Dataset:** [Flores-200](https://huggingface.co/datasets/facebook/flores) across 11 language pairs (en ↔ fr, es, pt, ja, zh, sw, wo, hi, ar, ru)