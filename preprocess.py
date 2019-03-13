import fire as fire

SPECIAL_CHAR = '~'


class Token:
    def __init__(self, conll_line):
        parts = conll_line.split('\t')
        self.index = int(parts[0])
        if parts[1] == SPECIAL_CHAR:
            parts[1] = '-'
        self.word = parts[1]
        self.lemma = parts[2]
        self.pos = parts[3]
        # self.xpos = parts[4]
        self.morphological_features = parts[5]

        # self.dep_root_index = int(parts[6])
        # self.dep_type = parts[7]

    def __str__(self):
        return self.word + ': ' + ','.join([self.index, self.word, self.lemma, self.pos, self.morphological_features])


class Sentence:
    def __init__(self, conll_lines):
        self.tokens = [Token(line) for line in conll_lines if line.split('\t')[0].isdigit()]

    def print_sentence(self):
        for token in self.tokens:
            print(token)


def preprocess(input_path, output_path):

    def token_format(t):
        return SPECIAL_CHAR.join(['w:' + t.word, 'm:' + t.morphological_features])

    with open(input_path, 'r') as rf, open(output_path, 'w') as wf:
        line = rf.readline()
        while line.strip() != '':
            conll_lines = []
            while line.strip() != '':
                conll_lines.append(line)
                line = rf.readline()
            if not conll_lines:
                line = rf.readline()
                continue
            sentence = Sentence(conll_lines)
            tokens = [t for t in sentence.tokens]
            wf.write(' '.join([token_format(t) for t in tokens]) + '\n')
            line = rf.readline()


if __name__ == "__main__":
    fire.Fire(preprocess)
