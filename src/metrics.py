from typing import Tuple, List

import numpy as np
from torchtext.data.metrics import bleu_score


def bleu_scorer(source: np.ndarray, predicted: np.ndarray, actual: np.ndarray, target_tokenizer, source_tokenizer):
    """Convert predictions to sentences and calculate
    BLEU score.

    Args:
        predicted (np.ndarray): batch of indices of predicted words
        actual (np.ndarray): batch of indices of ground truth words

    Returns:
        Tuple[float, List[str], List[str]]: tuple of
            (
                bleu score,
                ground truth sentences,
                predicted sentences
            )
    """
    batch_bleu = []
    predicted_sentences = []
    actual_sentences = []
    source_sentences = []
    for i, (a, b) in zip(source, zip(predicted, actual)):

        words_predicted = target_tokenizer.decode(a).split()
        words_actual = target_tokenizer.decode(b).split()
        words_source = source_tokenizer.decode(i).split()
        bls = bleu_score([words_predicted], [[words_actual]], max_n=4, weights=[0.25]*4)

        batch_bleu.append(bls)
        predicted_sentences.append(" ".join(words_predicted))
        actual_sentences.append(" ".join(words_actual))
        source_sentences.append(" ".join(words_source))
    batch_bleu = np.mean(batch_bleu)
    return batch_bleu, source_sentences, actual_sentences, predicted_sentences