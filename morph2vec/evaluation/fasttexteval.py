import warnings
from typing import Tuple, List, Union, Dict

import os
import fire
import numpy as np
import sklearn.utils

from fastText import load_model
from scipy import stats
from tqdm import tqdm


def load_eval_data(path: str):
    word_pairs: List[Tuple[str, str]] = []
    gold_similarity: List[float] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            w1, w2, sim = line.replace(',', ' ').split()
            word_pairs.append((w1, w2))
            gold_similarity.append(float(sim))

    return word_pairs, gold_similarity


def load_word_vectors(path: str) -> Dict[str, np.ndarray]:
    res = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w, v = line.strip().split(maxsplit=1)
            v = [float(i) for i in v.strip().split()]
            v = np.array(v)
            res[w] = v
    return res


def similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        warnings.warn('The norm of the vector is 0: returning similarity=0')
        return 0
    return np.dot(v1, v2) / n1 / n2


def evaluate_vectors(word_vectors: List[Tuple[np.ndarray, np.ndarray]], gold_similarity: List[float]) -> float:
    predicted_sim = []
    for v1, v2 in word_vectors:
        s = similarity(v1, v2)
        predicted_sim.append(s)

    assert len(predicted_sim) == len(gold_similarity)
    # The inspector thinks that stats.spearmanr takes integer arguments instead of lists
    # noinspection PyTypeChecker
    corr = stats.spearmanr(predicted_sim, gold_similarity)
    return corr[0]


def evaluate_model(model, word_pairs: List[Tuple[str, str]], gold_similarity: List[float],
                   with_vectors: bool = False) -> Union[float,
                                                        Tuple[float, List[Tuple[np.ndarray, np.ndarray]]]]:

    word_vectors = []
    for word1, word2 in word_pairs:
        v1, v2 = model.get_word_vector(word1), model.get_word_vector(word2)
        word_vectors.append((v1, v2))

    res = evaluate_vectors(word_vectors=word_vectors, gold_similarity=gold_similarity)
    if with_vectors:
        return res, word_vectors
    return res


def evaluate_cli(model_path: str, data_path: str, save_vectors_path: str = None):
    word_pairs, gold_similarity = load_eval_data(path=data_path)

    print('Loading fasttext model...', end=' ', flush=True)
    model = load_model(model_path)
    print('Done!')

    score, vectors = evaluate_model(model=model, word_pairs=word_pairs, gold_similarity=gold_similarity,
                                    with_vectors=True)
    dataset = os.path.basename(data_path)
    print(f'Score for the dataset {dataset}: {score:.3f}')

    assert len(vectors) == len(word_pairs), 'Need to have all vectors for all word-pairs'
    if not save_vectors_path:
        return
    with open(save_vectors_path, 'w', encoding='utf-8') as f:
        for (w1, w2), (v1, v2) in zip(word_pairs, vectors):
            f.write(w1 + '\t' + ' '.join([str(i) for i in v1.tolist()]) + '\n')
            f.write(w2 + '\t' + ' '.join([str(i) for i in v2.tolist()]) + '\n')


def bootstrap(word_pairs: List[Tuple[str, str]], gold_similarity: List[float],
              word2vec: Dict[str, np.ndarray] = None,
              bootstrap_count: int = 10000, confidence_percent: float = 0.95) -> Tuple[float, float, float, float]:

    scores = []
    for _ in tqdm(range(bootstrap_count)):
        cur_pairs, cur_gold = sklearn.utils.resample(word_pairs, gold_similarity, n_samples=len(word_pairs))
        score = evaluate_vectors(word_vectors=[(word2vec[w1], word2vec[w2]) for w1, w2 in cur_pairs],
                                 gold_similarity=cur_gold)
        scores.append(score)

    mean = np.mean(scores)
    std = np.std(scores)
    p = ((1.0 - confidence_percent) / 2.0) * 100
    lower = max(0.0, np.percentile(scores, p))
    p = (confidence_percent + ((1.0 - confidence_percent) / 2.0)) * 100
    upper = min(1.0, np.percentile(scores, p))

    return float(mean), float(std), lower, upper


def bootstrap_cli(gold_path: str, model_path: str = None, predicted_path: str = None,
                  bootstrap_count: int = 10000, confidence_percent: float = 0.95):

    word_pairs, gold_similarity = load_eval_data(path=gold_path)

    assert model_path or predicted_path, 'Either `model_path` or `predicted_path` need to be provided'
    if model_path:
        print('Loading fasttext model...', end=' ', flush=True)
        model = load_model(model_path)
        print('Done!')
        sim, predicted_word_vectors = evaluate_model(model=model,
                                                     word_pairs=word_pairs, gold_similarity=gold_similarity,
                                                     with_vectors=True)
        word2vec = {}
        for (w1, w2), (v1, v2) in zip(word_pairs, predicted_word_vectors):
            word2vec[w1] = v1
            word2vec[w2] = v2
    else:
        word2vec = load_word_vectors(path=predicted_path)

    mean, std, lower, upper = bootstrap(word2vec=word2vec,
                                        word_pairs=word_pairs, gold_similarity=gold_similarity,
                                        bootstrap_count=bootstrap_count,
                                        confidence_percent=confidence_percent)
    print(f'{mean:.3f} +/- {std:.3f} ({confidence_percent:.3f} confidence interval: [{lower:.3f}, {upper:.3f}])')


if __name__ == '__main__':
    fire.Fire({
        'evaluate': evaluate_cli,
        'bootstrap': bootstrap_cli,
    })
