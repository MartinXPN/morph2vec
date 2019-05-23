from pprint import pprint

import fire
from btb import HyperParameter, ParamTypes
from btb.tuning import GP
from fastText import train_unsupervised

from morph2vec.evaluation import fasttexteval


class HyperparameterSearchGym(object):
    def __init__(self, eval_train_path: str, eval_test_path: str):
        super(HyperparameterSearchGym, self).__init__()

        self.train_word_pairs, self.train_similarity = fasttexteval.load_eval_data(eval_train_path)
        self.test_word_pairs, self.test_similarity = fasttexteval.load_eval_data(eval_test_path)

        tunables = [
            ('lr',      HyperParameter(ParamTypes.FLOAT, [0.001, 0.1])),
            ('dim',     HyperParameter(ParamTypes.INT, [50, 350])),
            ('ws',      HyperParameter(ParamTypes.INT, [3, 11])),
            ('epoch',   HyperParameter(ParamTypes.INT, [3, 11])),
            ('minn',    HyperParameter(ParamTypes.INT, [2, 5])),
            ('maxn',    HyperParameter(ParamTypes.INT, [5, 9])),
            ('loss',    HyperParameter(ParamTypes.STRING, ['ns', 'hs'])),
        ]
        self.tuner = GP(tunables)

    def search_hyperparameters(self, nb_trials: int, input_path: str, props: str = 'w+l+t+m+n'):

        for trial in range(nb_trials):
            parameters = self.tuner.propose()
            pprint(parameters)

            ''' Construct and train the model '''
            model = train_unsupervised(input=input_path, props=props, **parameters)

            ''' Track results '''
            score = fasttexteval.evaluate_model(model=model,
                                                word_pairs=self.train_word_pairs,
                                                gold_similarity=self.train_similarity)
            self.tuner.add(parameters, score)
            print(f'Evaluation score: {score}')


if __name__ == '__main__':
    fire.Fire(HyperparameterSearchGym)
