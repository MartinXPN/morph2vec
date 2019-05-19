from typing import Tuple, List

import os
import fire
import numpy as np
import sklearn.utils

from fastText import load_model
from scipy import stats
from tqdm import tqdm


def similarity(v1: np.ndarray, v2: np.ndarray):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2


def evaluate(model, word_pairs: List[Tuple[str, str]], gold_similarity: List[float]):
    predicted_sim = []
    for word1, word2 in word_pairs:
        v1, v2 = model.get_word_vector(word1), model.get_word_vector(word2)
        assert np.any(v1), f'{word1} word-vector cannot be zero'
        assert np.any(v2), f'{word2} word-vector cannot be zero'
        s = similarity(v1, v2)
        predicted_sim.append(s)

    assert len(predicted_sim) == len(gold_similarity)
    # The inspector thinks that stats.spearmanr takes integer arguments instead of lists
    # noinspection PyTypeChecker
    corr = stats.spearmanr(predicted_sim, gold_similarity)
    return corr[0]


def load_eval_data(path: str):
    word_pairs: List[Tuple[str, str]] = []
    gold_similarity: List[float] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            w1, w2, sim = line.replace(',', ' ').split()
            word_pairs.append((w1, w2))
            gold_similarity.append(sim)

    return word_pairs, gold_similarity


def bootstrap(model, word_pairs: List[Tuple[str, str]], gold_similarity: List[float],
              bootrstrap_count: int, bootrstrap_split: float, confidence_percent: float = 0.95) -> Tuple[float, float]:

    scores = []
    for i in range(bootrstrap_count):
        cur_pairs, cur_gold = sklearn.utils.resample(word_pairs, gold_similarity,
                                                     n_samples=int(len(word_pairs) * bootrstrap_split))
        score = evaluate(model=model, word_pairs=cur_pairs, gold_similarity=cur_gold)
        scores.append(score)

    p = ((1.0 - confidence_percent) / 2.0) * 100
    lower = max(0.0, np.percentile(scores, p))
    p = (confidence_percent + ((1.0 - confidence_percent) / 2.0)) * 100
    upper = min(1.0, np.percentile(scores, p))

    return lower, upper


def main(model_path: str, data_path: str,
         bootstrap_count: int = 0, bootstrap_split: float = 0.8, confidence_percent: float = 0.95):
    word_pairs, gold_similarity = load_eval_data(path=data_path)

    print('Loading fasttext model...', end=' ', flush=True)
    model = load_model(model_path)
    print('Done!')

    if bootstrap_count == 0:
        score = evaluate(model=model, word_pairs=word_pairs, gold_similarity=gold_similarity)
        dataset = os.path.basename(data_path)
        print("Score for the dataset {0:20s}: {1:2.0f}".format(dataset, score * 100))

    else:
        lower, upper = bootstrap(model=model, word_pairs=word_pairs, gold_similarity=gold_similarity,
                                 bootrstrap_count=bootstrap_count, bootrstrap_split=bootstrap_split,
                                 confidence_percent=confidence_percent)
        print(f'{confidence_percent} confidence interval: [{lower}, {upper}]')


if __name__ == '__main__':
    fire.Fire(main)
