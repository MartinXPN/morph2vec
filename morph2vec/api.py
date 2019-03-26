from typing import Tuple, overload

from fastText import load_model
# noinspection PyProtectedMember
from fastText.FastText import _FastText as FastText

from morph2vec.data.preprocess import token_format
from morph2vec.entities.tokens import Token
from morph2vec.util.utils import download

BASE_URL = 'https://github.com/MartinXPN/morph2vec/releases/download'


class Morph2Vec(object):
    def __init__(self, model: FastText):
        self.model = model

    def __getitem__(self, item: str):
        return self.model.get_word_vector(word=item)

    def get_vector(self, word: str = '', lemma: str = '', pos: str = '',
                   morph_tags: Tuple[str] = tuple(), morphemes: Tuple[str] = tuple(), ngrams: Tuple[str] = tuple()):

        token = Token(index=0, word=word, lemma=lemma, pos=pos, xpos='',
                      morphological_tags=morph_tags, morphemes=morphemes, ngrams=ngrams)

        text_input = token_format(token)
        return self.model.get_word_vector(word=text_input)

    @classmethod
    @overload
    def load_model(cls, path: str) -> 'Morph2Vec':
        ...

    @classmethod
    @overload
    def load_model(cls, url: str, path: str) -> 'Morph2Vec':
        ...

    @classmethod
    @overload
    def load_model(cls, locale: str, version: str = None) -> 'Morph2Vec':
        ...

    @classmethod
    def load_model(cls, path: str = None, url: str = None, locale: str = None, version: str = None) -> 'Morph2Vec':
        from morph2vec import __version__

        if locale:
            version = version or __version__
            url = f'{BASE_URL}/v{version}/{locale}.bin'
            path = path or f'logs/{locale}-{version}.bin'

        if url and path:
            download(url, path, exists_ok=True)
        elif url:
            raise ValueError('Both URL and save path needs to be specified!')

        model = load_model(path=path)
        return cls(model)
