import os
import itertools

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14.tokenized.de-en.stanford/tmp'
LANG = 'de_tokensS'


def get_subword_tags(text_bpe):
    res = ''
    for (index_line, line_bpe) in enumerate(text_bpe.splitlines()):
        bpe_tokens = line_bpe.split()
        bpe_index = 0
        tag_index = 0
        res_line = ''
        while bpe_index < len(bpe_tokens):
            if '@@' in bpe_tokens[bpe_index]:
                first = True
                while '@@' in bpe_tokens[bpe_index]:
                    if first:
                        res_line += 'B' + ' '
                        first = False
                    else:
                        res_line += 'I' + ' '
                    bpe_index += 1
                    if '@@' not in bpe_tokens[bpe_index]:
                        res_line += 'E' + ' '
                        bpe_index += 1
                        tag_index += 1
                        break
            else:
                res_line += 'O' + ' '
                bpe_index += 1
                tag_index += 1
        res += res_line + '\n'
    return res


def main():
    for dataset in ['train', 'valid', 'test']:
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG), 'r') as file:
            text_bpe_tokens = file.read()
        subword_tags = get_subword_tags(text_bpe_tokens)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_subword_tags'), 'w') as file:
            file.write(subword_tags)


if __name__ == "__main__":
    main()
