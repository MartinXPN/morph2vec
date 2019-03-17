import re
from typing import Iterable, List, Tuple, Optional

import fire as fire
from nltk import everygrams
from word2morph.entities.sample import Sample
from word2morph.predict import Word2Morph

SPECIAL_CHAR = '~'


def parse(conll_lines: Iterable[str],
          min_ngram_len: int, max_ngram_len: int,
          word2morphemes: Optional[Word2Morph] = None):

    class Token:
        def __init__(self, conll_line: str):
            parts = conll_line.split('\t')
            self.index: int = int(parts[0])
            self.word: str = parts[1].replace(SPECIAL_CHAR, '-')
            self.lemma: str = parts[2].replace(SPECIAL_CHAR, '-')
            self.pos: str = parts[3]
            # self.xpos: str = parts[4]
            self.morphological_tags: List[str] = parts[5].split('|')
            self.morphemes: Tuple[str] = tuple()
            self.ngrams: List[str] = [''.join(g) for g in everygrams(self.word,
                                                                     min_len=min_ngram_len,
                                                                     max_len=max_ngram_len)]
            # self.dep_root_index: int = int(parts[6])
            # self.dep_type = parts[7]

    def token_format(t: Token):
        res = SPECIAL_CHAR.join([
            'w:' + t.word,
            'l:' + t.lemma,
            SPECIAL_CHAR.join(['t:' + str(mt) for mt in t.morphological_tags + ['POS=' + t.pos]]),
            SPECIAL_CHAR.join(['m:' + str(m) for m in t.morphemes]),
            SPECIAL_CHAR.join(['n:' + str(g) for g in t.ngrams]),
        ])
        # Replace consecutive duplicate SPECIAL_CHARs
        res = re.sub(r"{}+".format(SPECIAL_CHAR), SPECIAL_CHAR, res)
        if res[-1] == SPECIAL_CHAR:
            res = res[:-1]
        return res

    tokens = [Token(line) for line in conll_lines if line.split('\t')[0].isdigit()]

    ''' Add morpheme information (lemma2morph) '''
    if word2morphemes:
        morphemes = word2morphemes.predict(inputs=[Sample(word=tok.lemma, segments=tuple()) for tok in tokens],
                                           batch_size=1)
        for morph, tok in zip(morphemes, tokens):
            tok.morphemes = morph.segments
    return ' '.join([token_format(t) for t in tokens]) + '\n'


def preprocess(input_path: str, output_path: str,
               word2morphemes_model_path: str = None, word2morphemes_processor_path: str = None,
               min_ngram_len: int = 3, max_ngram_len: int = 6):

    print('Processing the file:', input_path)
    print('To save the results in:', output_path)
    word2morphemes = {} if word2morphemes_model_path is None or word2morphemes_processor_path is None \
        else Word2Morph(model_path=word2morphemes_model_path, processor_path=word2morphemes_processor_path)

    with open(input_path, 'r', encoding='utf-8') as rf, open(output_path, 'w', encoding='utf-8') as wf:
        line = rf.readline()
        while line.strip() != '':
            conll_lines = []
            while line.strip() != '':
                if line[0] != '#':
                    conll_lines.append(line)
                line = rf.readline()
            if not conll_lines:
                line = rf.readline()
                continue

            wf.write(parse(conll_lines=conll_lines, word2morphemes=word2morphemes,
                           min_ngram_len=min_ngram_len, max_ngram_len=max_ngram_len))
            line = rf.readline()


if __name__ == "__main__":
    fire.Fire(preprocess)
