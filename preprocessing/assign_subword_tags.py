import os

import itertools

BPE_TEXT_FILES_PATH = os.path.join('..','..','..','..','data','iwslt14-preprocessed-joined')
# PREPROCESSED_TEXT_FILES_PATH = os.path.join('..', '..', '..', '..', 'data', 'iwslt14-preprocessed-joined')
LANG = 'de'

def assign_subword_tags(text_bpe):
    BEGIN = 'B'
    INTERMEDIATE = 'I'
    END = 'E'
    ONLY_ONE = 'O'
    token_index = 0
    splitted_text_bpe = text_bpe.split()
    aligned_tags = ''
    end_of_lines_positions = list(itertools.accumulate(list(map(lambda x: len(x.split()), text_bpe.splitlines()))))
    end_of_line_index = 0
    while token_index < len(splitted_text_bpe):
        if '@@' in splitted_text_bpe[token_index]:
            first = True
            while '@@' in splitted_text_bpe[token_index]:
                if first:
                    aligned_tags += BEGIN
                    first = False
                else:
                    aligned_tags += INTERMEDIATE
                aligned_tags += ' '
                token_index += 1
                if '@@' not in splitted_text_bpe[token_index]:
                    aligned_tags += END
                    if token_index == end_of_lines_positions[end_of_line_index] - 1:
                        aligned_tags += '\n'
                        end_of_line_index += 1
                    else:
                        aligned_tags += ' '
                    token_index += 1
        else:
            aligned_tags += ONLY_ONE
            if token_index == end_of_lines_positions[end_of_line_index] - 1:
                aligned_tags += '\n'
                end_of_line_index += 1
            else:
                aligned_tags += ' '
            token_index += 1
    return aligned_tags


def main():
    for dataset in ['train', 'valid', 'test']:
        dataset_name = dataset + '.' + LANG
        with open(os.path.join(BPE_TEXT_FILES_PATH, dataset + '.bpe.' + LANG), 'r') as f:
            text_bpe = f.read()
        subword_tags = assign_subword_tags(text_bpe)
        with open(os.path.join(BPE_TEXT_FILES_PATH, dataset + '.bpe.' + LANG + '_subword_tags'), 'w') as f:
            f.write(subword_tags)


if __name__ == "__main__":
    main()

