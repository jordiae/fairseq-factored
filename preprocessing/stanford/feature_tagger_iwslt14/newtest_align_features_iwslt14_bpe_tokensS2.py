import os
import itertools

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt13.tokenized.de-en/tmp13'
LANG = 'de_tokensS'


def align_bpe(text_bpe, tags):
    res = ''
    for (index_line, (line_bpe, line_tags)) in enumerate(zip(text_bpe.splitlines(), tags.splitlines())):
        bpe_tokens = line_bpe.split()
        tag_tokens = line_tags.split()
        bpe_index = 0
        tag_index = 0
        res_line = ''
        while bpe_index < len(bpe_tokens):
            if '@@' in bpe_tokens[bpe_index]:
                while '@@' in bpe_tokens[bpe_index]:
                    res_line += tag_tokens[tag_index] + ' '
                    bpe_index += 1
                    if '@@' not in bpe_tokens[bpe_index]:
                        res_line += tag_tokens[tag_index] + ' '
                        bpe_index += 1
                        tag_index += 1
                        break
            else:
                res_line += tag_tokens[tag_index] + ' '
                bpe_index += 1
                tag_index += 1
        if tag_index != len(tag_tokens):
            print(res_line)
            print(line_bpe)
            print(line_tags)
            print(tag_index, len(tag_tokens))
            raise Exception('Ignored tags in line ' + str(index_line))
        res += res_line + '\n'
    return res


def main():
    for dataset in ['test']:
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG), 'r') as file:
            text_bpe_tokens = file.read()
        with open(os.path.join(PATH, dataset + '.' + 'de' + '_lemmasS'), 'r') as file:
            text_lemma = file.read()
        repeated_lemmas = align_bpe(text_bpe_tokens, text_lemma)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_lemmas'), 'w') as file:
            file.write(repeated_lemmas)


if __name__ == "__main__":
    main()
