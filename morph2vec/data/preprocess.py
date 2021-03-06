import re
from typing import List, Dict

import fire as fire
from sentence2tags import Sentence2Tags, tree_to_conllu_lines, sentence_to_tree
from tqdm import tqdm
from word2morph import Word2Morph
from word2morph.entities.sample import Sample

from morph2vec.entities.tokens import Token, TokenFactory

SPECIAL_CHAR = '~'


def token_format(t: Token):
    res = SPECIAL_CHAR.join([
        'w:' + t.word,
        'l:' + t.lemma,
        SPECIAL_CHAR.join(['t:' + str(mt) for mt in t.morphological_tags + ('POS=' + t.pos,)]),
        SPECIAL_CHAR.join(['m:' + str(m) for m in t.morphemes]),
        SPECIAL_CHAR.join(['n:' + str(g) for g in t.ngrams]),
    ])
    # Replace consecutive duplicate SPECIAL_CHARs
    res = re.sub(r"{}+".format(SPECIAL_CHAR), SPECIAL_CHAR, res)
    if res[-1] == SPECIAL_CHAR:
        res = res[:-1]
    return res


def preprocess_conllu(input_path: str, output_path: str, locale: str = None,
                      min_ngram_len: int = 3, max_ngram_len: int = 6):
    """ Preprocess the data which is in the CONNL-U format """

    print('Processing the file:', input_path)
    print('To save the results in:', output_path)
    word2morphemes = {} if locale is None else Word2Morph.load_model(locale=locale)

    sentences = []
    with open(input_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line.strip() != '':
            conll_lines = []
            while line.strip() != '':
                if line.split('\t')[0].isdigit():
                    conll_lines.append(line)
                line = f.readline()
            if not conll_lines:
                line = f.readline()
                continue

            sentences.append(conll_lines)
            line = f.readline()

    print('Processing', len(sentences), 'sentences...', flush=True)
    get_token = TokenFactory(special_char=SPECIAL_CHAR, word2morphemes=word2morphemes,
                             min_ngram_len=min_ngram_len, max_ngram_len=max_ngram_len)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in tqdm(sentences):
            tokens = [get_token.from_conll_line(line) for line in sentence]
            parsed_sentence = ' '.join([token_format(t) for t in tokens])
            f.write(parsed_sentence + '\n')


def preprocess_wiki(input_path: str, output_path: str, locale: str,
                    min_ngram_len: int = 3, max_ngram_len: int = 6, max_sentence_len: int = 100,
                    chunk_size: int = 10, batch_size: int = 32):
    """
    Preprocess wiki. As the file is too large (5+GB) there are some optimisations made here.
    We keep a cache of words which were already processed by word2morph not to process the same word twice.
    The sentences are processed in batches rather than one by one to fully utilise the GPUs available.
    We process `chunk_size` sentences at once giving `batch_size` elements to the neural network for each batch in the
    chunk.
    """
    def lemmas_to_morph(lemmas: List[str]) -> List[Sample]:

        absent_lemmas = [lemma for lemma in lemmas if lemma not in lemmas_to_morph.cache]
        input_lemmas = [Sample(word=lemma, segments=tuple()) for lemma in absent_lemmas]
        res_samples = word2morphemes.predict(inputs=input_lemmas, batch_size=batch_size)

        for lemma, sample in zip(absent_lemmas, res_samples):
            lemmas_to_morph.cache[lemma] = sample

        return [lemmas_to_morph.cache[lemma] for lemma in lemmas]

    lemmas_to_morph.cache: Dict[str, Sample] = {}

    def process(chunk_sentences: List[str]):
        """ Process chunk os sentences returning result in the needed format """
        ''' Sentence to tags '''
        input_trees = [sentence_to_tree(s.split(' ')) for s in chunk_sentences if len(s.split(' ')) < max_sentence_len]
        res_trees = sentence2tags.predict(input_trees)

        ''' Parse all the sentences (trees) to CONLL-U format '''
        res_sentences_conllu = [tree_to_conllu_lines(t) for t in res_trees]

        ''' Get all the tokens from the CONLL-U formatted sentences  '''
        res_sentences_tokens = [[get_token.from_conll_line(line) for line in sentence_conllu]
                                for sentence_conllu in res_sentences_conllu]

        ''' Parse lemmas to get morphemes here instead of get_token.from_conll_line '''
        lemmas = [[t.lemma for t in sentence_tokens] for sentence_tokens in res_sentences_tokens]
        all_lemmas = [l for li in lemmas for l in li]
        all_morphemes = lemmas_to_morph(lemmas=all_lemmas)

        ''' Add morphemes to the tokens '''
        idx = 0
        for sentence_tokens in res_sentences_tokens:
            for t in sentence_tokens:
                t.morphemes = all_morphemes[idx].segments
                idx += 1

        parsed_sentences = [' '.join([token_format(t) for t in sentence_tokens])
                            for sentence_tokens in res_sentences_tokens]
        return parsed_sentences

    with open(input_path, 'r', encoding='utf-8') as f:
        sentences = [l.strip() for l in tqdm(f)]

    word2morphemes = Word2Morph.load_model(locale=locale)
    sentence2tags = Sentence2Tags.load_model(locale=locale)
    get_token = TokenFactory(special_char=SPECIAL_CHAR, min_ngram_len=min_ngram_len, max_ngram_len=max_ngram_len)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(0, len(sentences), chunk_size)):
            for p in process(sentences[i: i + chunk_size]):
                f.write(p + '\n')


def preprocess_eval(input_path: str, output_path: str, locale: str,
                    min_ngram_len: int = 3, max_ngram_len: int = 6):
    """ Preprocess the evalution data where the file is of the format `word1, word2, similarity` """

    word2morphemes = Word2Morph.load_model(locale=locale)
    sentence2tags = Sentence2Tags.load_model(locale=locale)
    get_token = TokenFactory(special_char=SPECIAL_CHAR,
                             word2morphemes=word2morphemes, sentence2tags=sentence2tags,
                             min_ngram_len=min_ngram_len, max_ngram_len=max_ngram_len)

    with open(output_path, 'w', encoding='utf-8') as outf, open(input_path, 'r', encoding='utf-8') as inf:
        for line in tqdm(inf):
            w1, w2, sim = line.replace(',', ' ').split()
            w1_token = get_token.from_word(w1)
            w2_token = get_token.from_word(w2)

            outf.write(token_format(w1_token) + ' ' +
                       token_format(w2_token) + ' ' + str(sim) + '\n')


if __name__ == "__main__":
    fire.Fire()
