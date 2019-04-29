# morph2vec

This repo is a modification of [prop2vec](https://github.com/oavraham1/prop2vec) code 
with some additional features on top of it.

#### Current Implementation:
* fastText to get word-vectors
* preprocessing step to modify the `.conllu` file to fit it to the fastText model

#### Modifications and additional features
* added support for WLTMN input
    * W: wordform
    * L: lemma
    * T: morphological tags
    * M: morphemes
    * N: n-grams
* morphemes are extracted with [word2morph](https://github.com/MartinXPN/word2morph)
which takes longer time because a neural network is used to extract
morphemes from a given lemma
* n-grams are added during the preprocessing step (not fasttext default one)

## Instructions
* To download sample data in `.conllu` format:
```commandline
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Armenian-ArmTDP/master/hy_armtdp-ud-train.conllu -P datasets
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-train.conllu -P datasets
```

* To prepare the data for training a fastText model (the last two arguments are optional):
```commandline
# Preprocess the evaluation data (if needed)
PYTHONHASHSEED=0 python -m morph2vec.data preprocess_eval --input_path datasets/eval-train.txt --output_path datasets/eval-train-processed.txt --locale ru
# Preprocess the conllu corpus
PYTHONHASHSEED=0 python -m morph2vec.data.preprocess preprocess_conllu --input_path datasets/ru_syntagrus-ud-train.conllu --output_path datasets/ru_processed_wltmn.txt  --locale ru
```

* To train a fastText model:
```commandline
PYTHONHASHSEED=0 python -m morph2vec.train 
        train_unsupervised --input datasets/ru_processed_wltmn.txt --model skipgram --props w+l+t+m+n --lr 0.025 --dim 200 --ws 2 --epoch 5 --minCount 5 --minCountLabel 0 --minn 3 --maxn 6 --neg 5 --wordNgrams 1 --loss ns --bucket 2000000 --thread 1 --lrUpdateRate 100 --t 1e-3 --label __label__ --verbose 2 --pretrainedVectors ""
        save_model --path logs/ru.bin
```

* To do a hyperparameter search:
```commandline
PYTHONHASHSEED=0 python -m morph2vec.hyperparametersearch
        --eval_train_path datasets/eval-train-processed.txt --eval_test_path datasets/eval-test-processed.txt
        search_hyperparameters --nb_trials 500 --input_path datasets/ru_processed_wltmn.txt --props "w+l+t+m+n"
```
