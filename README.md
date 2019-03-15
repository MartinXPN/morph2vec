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

#### To run the process
```commandline
./run.sh
```

