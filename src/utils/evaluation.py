from sacrebleu.metrics import BLEU, CHRF
import transformers
from datasets import Dataset

bleu = BLEU(tokenize="flores200", effective_order=True)
chrf = CHRF(word_order=2)


def eval_bleu(reference: str, generation: str) -> float:
    return bleu.sentence_score(generation, [reference]).score


def eval_chrf(reference: str, generation: str) -> float:
    return chrf.sentence_score(generation, [reference]).score
