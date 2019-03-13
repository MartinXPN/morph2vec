import re

import fire as fire
from nltk import everygrams

SPECIAL_CHAR = '~'


class Token:
    def __init__(self, conll_line, min_ngram_len, max_ngram_len):
        parts = conll_line.split('\t')
        self.index = int(parts[0])
        self.word = parts[1].replace(SPECIAL_CHAR, '-')
        self.lemma = parts[2].replace(SPECIAL_CHAR, '-')
        self.pos = parts[3]
        # self.xpos = parts[4]
        self.morphological_tags = parts[5].split('|')
        self.morphemes = []
        self.ngrams = [''.join(g) for g in everygrams(self.word, min_len=min_ngram_len, max_len=max_ngram_len)]
        # self.dep_root_index = int(parts[6])
        # self.dep_type = parts[7]


class Sentence:
    def __init__(self, conll_lines, min_ngram_len, max_ngram_len):
        self.tokens = [Token(line, min_ngram_len, max_ngram_len)
                       for line in conll_lines
                       if line.split('\t')[0].isdigit()]


def preprocess(input_path, output_path, min_ngram_len=3, max_ngram_len=6):
    print('Input path:', input_path)
    print('Output path:', output_path)

    def token_format(t):
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

    with open(input_path, 'r', encoding='utf-8') as rf, open(output_path, 'w', encoding='utf-8') as wf:
        line = rf.readline()
        while line.strip() != '':
            conll_lines = []
            while line.strip() != '':
                conll_lines.append(line)
                line = rf.readline()
            if not conll_lines:
                line = rf.readline()
                continue
            sentence = Sentence(conll_lines, min_ngram_len, max_ngram_len)
            tokens = [t for t in sentence.tokens]
            wf.write(' '.join([token_format(t) for t in tokens]) + '\n')
            line = rf.readline()


if __name__ == "__main__":
    fire.Fire(preprocess)
