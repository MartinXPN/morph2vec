import urllib.request
from pathlib import Path
from typing import Tuple

from fastText import load_model
# noinspection PyProtectedMember
from fastText.FastText import _FastText as FastText

from morph2vec.data.preprocess import token_format
from morph2vec.entities.tokens import Token
from morph2vec.util.utils import DownloadProgressBar

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

    @staticmethod
    def load_model(path: str = None, url: str = None, locale: str = None, version: str = None) -> 'Morph2Vec':
        from morph2vec import __version__

        if locale:
            version = version or __version__
            url = '{BASE_URL}/v{version}/{locale}.bin'.format(BASE_URL=BASE_URL, version=version, locale=locale)
            path = path or 'logs/{locale}-{version}.bin'.format(locale=locale, version=version)

        if url and path:
            if not Path(path).exists():
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url=url, filename=path, reporthook=t.update_to)
            else:
                print('Model already exists. Loading an existing file...')
        elif url:
            raise ValueError('Both URL and save path needs to be specified!')

        model = load_model(path=path)
        return Morph2Vec(model)
