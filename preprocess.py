import json
import re
from typing import Iterable, Dict, Tuple

import fire as fire
from nltk import everygrams

SPECIAL_CHAR = '~'


def parse(conll_lines: Iterable[str], word2morphemes: Dict[str, Tuple[str]], min_ngram_len: int, max_ngram_len: int):

    class Token:
        def __init__(self, conll_line: str):
            parts = conll_line.split('\t')
            self.index = int(parts[0])
            self.word = parts[1].replace(SPECIAL_CHAR, '-')
            self.lemma = parts[2].replace(SPECIAL_CHAR, '-')
            self.pos = parts[3]
            # self.xpos = parts[4]
            self.morphological_tags = parts[5].split('|')
            self.morphemes = word2morphemes[self.lemma] if self.lemma in word2morphemes else tuple()
            self.ngrams = [''.join(g) for g in everygrams(self.word, min_len=min_ngram_len, max_len=max_ngram_len)]
            # self.dep_root_index = int(parts[6])
            # self.dep_type = parts[7]

    def token_format(t: Token):
        res = SPECIAL_CHAR.join([
            'w:' + t.word,
            'l:' + t.lemma,
            SPECIAL_CHAR.join(['t:' + mt for mt in t.morphological_tags + ['POS=' + t.pos]]),
            SPECIAL_CHAR.join(['m:' + m for m in t.morphemes]),
            SPECIAL_CHAR.join(['n:' + g for g in t.ngrams]),
        ])
        # Replace consecutive duplicate SPECIAL_CHARs
        res = re.sub(r"{}+".format(SPECIAL_CHAR), SPECIAL_CHAR, res)
        if res[-1] == SPECIAL_CHAR:
            res = res[:-1]
        return res

    tokens = [Token(line) for line in conll_lines if line.split('\t')[0].isdigit()]
    return ' '.join([token_format(t) for t in tokens]) + '\n'


def preprocess(input_path: str, output_path: str, word2morphemes_path: str = None,
               min_ngram_len: int = 3, max_ngram_len: int = 6):

    word2morphemes = json.load(word2morphemes_path) if word2morphemes_path else dict()

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
