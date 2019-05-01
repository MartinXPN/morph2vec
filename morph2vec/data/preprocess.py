import re

import fire as fire
from sentence2tags import Sentence2Tags, tree_to_conllu_lines, sentence_to_tree
from tqdm import tqdm
from word2morph import Word2Morph

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
                    min_ngram_len: int = 3, max_ngram_len: int = 6):
    with open(input_path, 'r', encoding='utf-8') as f:
        sentences = [l.strip() for l in tqdm(f)]

    word2morphemes = Word2Morph.load_model(locale=locale)
    sentence2tags = Sentence2Tags.load_model(locale=locale)
    get_token = TokenFactory(special_char=SPECIAL_CHAR, word2morphemes=word2morphemes,
                             min_ngram_len=min_ngram_len, max_ngram_len=max_ngram_len)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in tqdm(sentences):
            input_tree = sentence_to_tree(sentence.split(' '))
            res_tree = sentence2tags[input_tree]
            conllu_lines = tree_to_conllu_lines(res_tree)

            tokens = [get_token.from_conll_line(line) for line in conllu_lines]
            parsed_sentence = ' '.join([token_format(t) for t in tokens])
            f.write(parsed_sentence + '\n')


def preprocess_eval(input_path: str, output_path: str, locale: str,
                    min_ngram_len: int = 3, max_ngram_len: int = 6):
    print('Processing the file:', input_path)
    print('To save the results in:', output_path)
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
