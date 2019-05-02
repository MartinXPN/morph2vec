from typing import Tuple, Dict

from nltk import everygrams
from sentence2tags import Sentence2Tags
from sentence2tags.api import sentence_to_tree


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
    def __init__(self, special_char: str, word2morphemes=None, sentence2tags: Sentence2Tags = None,
                 min_ngram_len: int = 3, max_ngram_len: int = 6):
        self.special_char = special_char
        self.word2morphemes = word2morphemes
        self.sentence2tags = sentence2tags
        self.min_ngram_len = min_ngram_len
        self.max_ngram_len = max_ngram_len
        self.cache: Dict[str, Tuple[str, ...]] = {}

    def from_conll_line(self, line):
        parts = line.split('\t')
        word = parts[1].replace(self.special_char, '-')
        lemma = parts[2].replace(self.special_char, '-').lower()

        if lemma in self.cache:
            morphemes = self.cache[lemma]
        else:
            morphemes = self.cache[lemma] = self.word2morphemes[lemma].segments \
                if self.word2morphemes and lemma else tuple()

        return Token(index=int(parts[0]),
                     word=word,
                     lemma=lemma,
                     pos=parts[3],
                     xpos=parts[4],
                     morphological_tags=tuple(parts[5].split('|')),
                     morphemes=morphemes,
                     ngrams=tuple([''.join(g) for g in everygrams(word,
                                                                  min_len=self.min_ngram_len,
                                                                  max_len=self.max_ngram_len)]),
                     )

    def from_word(self, word):
        tree = sentence_to_tree(sentence=[word])
        word_token = self.sentence2tags[tree].tokens[1]

        lemma = word_token.fields['lemma'].lower()
        pos_tag = word_token.fields['upostag']
        morph_tags = word_token.fields['feats']

        # index, word-form, lemma, pos, xpos, morph-tags
        conll_line = f'{1}\t{word}\t{lemma}\t{pos_tag}\t{pos_tag}\t{morph_tags}'
        return self.from_conll_line(line=conll_line)
