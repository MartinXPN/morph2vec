import re
from typing import Iterable, List, Tuple, Optional

import fire as fire
from nltk import everygrams
from tqdm import tqdm
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
            self.morphemes: Tuple[str] = word2morphemes[self.lemma].segments if word2morphemes else tuple()
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
    return ' '.join([token_format(t) for t in tokens]) + '\n'


def preprocess(input_path: str, output_path: str,
               word2morphemes_model_path: str = None, word2morphemes_processor_path: str = None,
               min_ngram_len: int = 3, max_ngram_len: int = 6):

    print('Processing the file:', input_path)
    print('To save the results in:', output_path)
    word2morphemes = {} if word2morphemes_model_path is None or word2morphemes_processor_path is None \
        else Word2Morph(model_path=word2morphemes_model_path, processor_path=word2morphemes_processor_path)

    sentences = []
    with open(input_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line.strip() != '':
            conll_lines = []
            while line.strip() != '':
                if line[0] != '#':
                    conll_lines.append(line)
                line = f.readline()
            if not conll_lines:
                line = f.readline()
                continue

            sentences.append(conll_lines)
            line = f.readline()

    print('Processing', len(sentences), 'sentences...', flush=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in tqdm(sentences):
            f.write(parse(conll_lines=sentence, word2morphemes=word2morphemes,
                          min_ngram_len=min_ngram_len, max_ngram_len=max_ngram_len))


if __name__ == "__main__":
    fire.Fire(preprocess)
