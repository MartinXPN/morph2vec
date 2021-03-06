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
```bash
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Armenian-ArmTDP/master/hy_armtdp-ud-train.conllu -P datasets
wget https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-train.conllu -P datasets
```

* To download the preprocessed wiki corpus:
```bash
wget https://github.com/MartinXPN/morph2vec/releases/download/v0.2.0/ru-wiki-text.zip -P datasets
unzip ru-wiki-text.zip
```

* The evaluation data can be obtained from [russe-evaluation](https://github.com/nlpub/russe-evaluation/blob/master/russe/evaluation/README.md) corpus
* To prepare the data for training a fastText model:
```bash
# [Optional] Preprocess the evaluation data
PYTHONHASHSEED=0 python -m morph2vec.data.preprocess preprocess_eval --input_path datasets/eval-train.txt --output_path datasets/eval-train-processed.txt --locale ru
PYTHONHASHSEED=0 python -m morph2vec.data.preprocess preprocess_eval --input_path datasets/eval-test.txt --output_path datasets/eval-test-processed.txt --locale ru

# Preprocess the conllu corpus
PYTHONHASHSEED=0 python -m morph2vec.data.preprocess preprocess_conllu --input_path datasets/ru_syntagrus-ud-train.conllu --output_path datasets/ru_processed_wltmn.txt  --locale ru

# Preprocess the wiki corpus
PYTHONHASHSEED=0 python -m morph2vec.data.preprocess preprocess_wiki datasets/ru-wiki-text.txt --output_path datasets/ru-wiki.wltmn --locale ru
```

* To train a fastText model (Training a model on half of the russian wiki takes ~4 hours on 4 core CPU):
```bash
PYTHONHASHSEED=0 python -m morph2vec.train 
        train_unsupervised --input datasets/ru-wiki.wltmn --model skipgram --props w+l+t+m --lr 0.05 --dim 300 --ws 5 --epoch 5 --minCount 5 --minCountLabel 0 --minn 3 --maxn 6 --neg 5 --wordNgrams 1 --loss ns --bucket 2000000 --thread 15 --lrUpdateRate 100 --t 1e-4 --label __label__ --verbose 2 --pretrainedVectors "" 
        save_model --path logs/ru-wltm.bin
```

* To do a hyperparameter search:
```bash
PYTHONHASHSEED=0 python -m morph2vec.hyperparametersearch
        --eval_train_path datasets/eval-train-processed.txt --eval_test_path datasets/eval-test-processed.txt
        search_hyperparameters --nb_trials 500 --input_path datasets/ru_processed_wltmn.txt --props "w+l+t+m+n"
```

* To evaluate the model:
```bash
# Evaluate and save results:
PYTHONHASHSEED=0 python -m morph2vec.evaluation.fasttexteval evaluate --model_path logs/ru.bin --data_path datasets/hj-expanded-rare-all-processed.txt --save_vectors_path logs/hj-pred.txt


# Bootstrapping to get confidence intervals, mean and std of Spearman's correlation
PYTHONHASHSEED=0 python -m morph2vec.evaluation.fasttexteval bootstrap --gold_path datasets/hj-expanded-rare-all-processed.txt --predicted_path logs/hj-pred.txt --bootstrap_count 10000 --confidence_percent 0.95

# Or by providing a model instead of the (word  vector) file
PYTHONHASHSEED=0 python -m morph2vec.evaluation.fasttexteval bootstrap --gold_path datasets/hj-expanded-rare-all-processed.txt --model_path logs/ru.bin --bootstrap_count 10000 --confidence_percent 0.95
```