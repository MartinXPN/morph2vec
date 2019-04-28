from typing import Tuple, List

import fire
import numpy as np
from fastText import load_model
from scipy import stats
import os


def similarity(v1: np.ndarray, v2: np.ndarray):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2


def evaluate(model, word_pairs: List[Tuple[str, str]], gold_similarity: List[float]):
    predicted_sim = []
    for word1, word2 in word_pairs:
        v1, v2 = model.get_word_vector(word1), model.get_word_vector(word2)
        s = similarity(v1, v2)
        predicted_sim.append(s)

    # The inspector thinks that stats.spearmanr takes integer arguments instead of lists
    # noinspection PyTypeChecker
    corr = stats.spearmanr(predicted_sim, gold_similarity)
    return corr[0]


def main(model_path: str, data_path: str):
    print('Loading the data...', flush=True)
    word_pairs = []
    gold_similarity = []
    with open(data_path, 'r', encoding='utf-8') as f:
        w1, w2, sim = f.readline().replace(',', ' ').split()
        word_pairs.append((w1, w2))
        gold_similarity.append(sim)

    print('Loading fasttext model...', end=' ', flush=True)
    model = load_model(model_path)
    print('Done!')

    score = evaluate(model=model, word_pairs=word_pairs, gold_similarity=gold_similarity)
    dataset = os.path.basename(data_path)
    print("{0:20s}: {1:2.0f}".format(dataset, score * 100))


if __name__ == '__main__':
    fire.Fire(main)
