from typing import Tuple

from nltk import everygrams


class Token(object):
    def __init__(self,
                 index: int, word: str, lemma: str, pos: str, xpos: str,
                 morphological_tags: Tuple[str, ...], morphemes: Tuple[str, ...], ngrams: Tuple[str, ...]):
        self.index = index
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.xpos = xpos
        self.morphological_tags = morphological_tags
        self.morphemes = morphemes
        self.ngrams = ngrams


class TokenFactory(object):
    def __init__(self, special_char: str, word2morphemes=None, min_ngram_len: int = 3, max_ngram_len: int = 6):
        self.special_char = special_char
        self.word2morphemes = word2morphemes
        self.min_ngram_len = min_ngram_len
        self.max_ngram_len = max_ngram_len

    def from_conll_line(self, line):
        parts = line.split('\t')
        word = parts[1].replace(self.special_char, '-')
        lemma = parts[2].replace(self.special_char, '-')
        return Token(index=int(parts[0]),
                     word=word,
                     lemma=lemma,
                     pos=parts[3],
                     xpos=parts[4],
                     morphological_tags=tuple(parts[5].split('|')),
                     morphemes=self.word2morphemes[lemma].segments if self.word2morphemes else tuple(),
                     ngrams=tuple([''.join(g) for g in everygrams(word,
                                                                  min_len=self.min_ngram_len,
                                                                  max_len=self.max_ngram_len)]),
                     )
