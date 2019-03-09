import getopt
import inspect
import os
import sys


class Token:
    def __init__(self, conll_line):
        parts = conll_line.split('\t')
        self.index = int(parts[0])
        if parts[1] == special_char:
            parts[1] = '-'
        self.word = parts[1]
        self.dep_root_index = int(parts[6])
        self.dep_type = parts[7]
        self.morph = self.__get_morph(parts)

    def __str__(self):
        return self.word + ': ' + ','.join([self.index, self.morph, self.dep_type])

    def __get_morph(self, parts):
        pos = parts[3]
        detailed_pos = parts[4]

        if detailed_pos == "VB-TOINFINITIVE":
            morph = "tense=TOINFINITIVE"
        else:
            morph = parts[5]
        if morph == '_':
            return pos
        if pos == 'BN':  # unify present tense
            pos = 'VB'
            morph += "|tense=BEINONI"
        morph_parts = morph.split('|')
        if len(morph_parts) == 5:
            morph_parts = ['gen=MF'] + morph_parts[2:]  # fix 'MF' gender format
        morph = '-'.join([pos] + [p.split('=')[1] for p in morph_parts])
        if 'POS' in detailed_pos:  # unify possessives
            morph += '-B'

        return morph


class Sentence:
    def __init__(self, conll_lines):
        self.tokens = [Token(line) for line in conll_lines if line.split('\t')[0].isdigit()]

    def print_sentence(self):
        for token in self.tokens:
            print(token)


special_char = '~'


class ModelCreator:
    def __init__(self, should_filter, token_format):
        self.should_filter = should_filter
        self.token_format = token_format


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir + '/utils')
    import utils
    word2count = utils.get_word2count(inputfile)

    def get_count(word):
        if word not in word2count:
            return 0
        return word2count[word]

    def is_marker(word):
        return word in ['*PRP*', '*POS*', '*DEF*', '*ACC*']

    should_filter = lambda word: get_count(word) < 5 or is_marker(word)
    token_format = lambda t: special_char.join(['w:' + t.word, 'm:' + t.morph])

    with open(inputfile, 'rb') as rf, open(outputfile, 'w') as wf:
        line = rf.readline()
        while line != '':
            conll_lines = []
            while line.strip() != '':
                conll_lines.append(line)
                line = rf.readline()
            if not conll_lines:
                line = rf.readline()
                continue
            sentence = Sentence(conll_lines)
            tokens = [t for t in sentence.tokens if not should_filter(t.word)]
            wf.write(' '.join([token_format(t) for t in tokens]) + '\n')
            line = rf.readline()


if __name__ == "__main__":
    main(sys.argv[1:])
