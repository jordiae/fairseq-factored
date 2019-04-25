import os

import datetime

from ast import literal_eval

import re

import itertools

TOKENIZED_TEXT_FILES_PATH = os.path.join('..', '..', '..', '..', 'data', 'iwslt14.tokenized.de-en', 'tmp')
BPE_TEXT_FILES_PATH = os.path.join('..','..','..','..','data','iwslt14-preprocessed-joined')
# PREPROCESSED_TEXT_FILES_PATH = os.path.join('..', '..', '..', '..', 'data', 'iwslt14-preprocessed-joined')
LANG = 'de'
CHAR_LIMIT = 3500


def get_chunks(s, n_chars):
    chunks = []
    current_chunk = ''
    current_char_count = 0
    for line in s.splitlines():
        new_count = len(line) + 1 + current_char_count
        if new_count <= n_chars:
            current_chunk = current_chunk + line + '\n'
            current_char_count = new_count
        else:
            chunks.append(current_chunk)
            current_chunk = line + '\n'
            current_char_count = len(line) + 1
    if len(current_chunk) > 0:
        chunks.append(current_chunk)
    return chunks


def align_indices(text_chunks, parsed_chunks):
    cumulative_index = 0
    for text_chunk, parsed_chunk in zip(text_chunks, parsed_chunks):
        for synset in parsed_chunk:
            synset[1] += cumulative_index
            synset[2] += cumulative_index
        cumulative_index += len(text_chunk)
    return parsed_chunks

# Priority: multi-word concepts.
def assign_synsets(synsets, text):
    indices_split = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
    assigned_synsets = [None] * len(indices_split)
    index_dict = dict(zip([a for a, b in indices_split], list(range(0, len(indices_split)))))
    synset_dict = {}
    i = -1
    for synset, start_synset, end_synset in synsets:
        i += 1
        end_synset += 1  # Babelfy synsets have end of word as subset, unlike Python slices/indices.
        try:
            word_index = index_dict[start_synset]
        except:  # if non-alphanumeric chars break tokenization at the beginning of the word
            retry_count = 0
            start_synset -= 1
            max_retry = 5
            while retry_count < max_retry:
                retry_count += 1
                try:
                    word_index = index_dict[start_synset]
                    break
                except:
                    start_synset -= 1
                    continue
            if retry_count == max_retry:
                raise Exception('Malformed token at the beginning of the word!', start_synset, i)
        start_word, end_word = indices_split[word_index]
        if start_word == start_synset and end_word == end_synset:
            if assigned_synsets[word_index] is None:
                assigned_synsets[word_index] = [synset]
            else:
                assigned_synsets[word_index].append(synset)
            synset_dict[synset] = 1
        elif start_word == start_synset and end_word < end_synset:
            if assigned_synsets[word_index] is None:
                assigned_synsets[word_index] = [synset]
            else:
                assigned_synsets[word_index].append(synset)
            word_index += 1
            start_word, end_word = indices_split[word_index]
            synset_count = 1
            while end_word <= end_synset:
                if assigned_synsets[word_index] is None:
                    assigned_synsets[word_index] = [synset]
                else:
                    assigned_synsets[word_index].append(synset)
                synset_count += 1
                word_index += 1
                start_word, end_word = indices_split[word_index]
            synset_dict[synset] = synset_count
        elif start_word == start_synset:  # if non-alphanumeric chars break tokenization at the end of the word
            start_word2, end_word2 = indices_split[word_index+1]
            if start_word2 > end_synset:
                if assigned_synsets[word_index] is None:
                    assigned_synsets[word_index] = [synset]
                else:
                    assigned_synsets[word_index].append(synset)
                synset_dict[synset] = 1
            else:
                print(synset, start_synset, end_synset, word_index)
                raise Exception('Discarded synset (malformed offset at the beginning)!')
        else:
            print(synset, start_synset, end_synset, word_index)
            raise Exception('Discarded synset!')
    result = []
    for index, token in enumerate(text.split()):
        if assigned_synsets[index] is None:
            result.append(None)
        else:
            max = -1
            chosen_synset = None
            for synset in assigned_synsets[index]:
                if synset_dict[synset] > max:
                    max = synset_dict[synset]
                    chosen_synset = synset
            result.append(chosen_synset)
    return result



def flatten(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


def assign_POS_to_unknown_synsets(synsets, POS_path):
    pos_text = None
    with open(POS_path, 'r') as file:
        pos_text = file.read()
    pos_tags = pos_text.split()
    i = 0
    for synset, pos_tag in zip(synsets, pos_tags):
        if synset is None:
            synsets[i] = pos_tag
        i += 1
    return synsets


def align_synsets_bpe(synsets, text_bpe):
    token_index = 0
    tag_index = 0
    splitted_text_bpe = text_bpe.split()
    splitted_synsets = synsets
    aligned_synsets = ''
    end_of_lines_positions = list(itertools.accumulate(list(map(lambda x: len(x.split()), text_bpe.splitlines()))))
    end_of_line_index = 0
    while token_index < len(splitted_text_bpe):
        if '@@' in splitted_text_bpe[token_index]:
            while '@@' in splitted_text_bpe[token_index]:
                aligned_synsets += (splitted_synsets[tag_index])
                aligned_synsets += '@@'
                aligned_synsets += ' '
                token_index += 1
                if '@@' not in splitted_text_bpe[token_index]:
                    aligned_synsets += (splitted_synsets[tag_index])
                    if token_index == end_of_lines_positions[end_of_line_index] - 1:
                        aligned_synsets += '\n'
                        end_of_line_index += 1
                    else:
                        aligned_synsets += ' '
                    token_index += 1
                    tag_index += 1
        else:
            aligned_synsets += (splitted_synsets[tag_index])
            if token_index == end_of_lines_positions[end_of_line_index] - 1:
                aligned_synsets += '\n'
                end_of_line_index += 1
            else:
                aligned_synsets += ' '
            token_index += 1
            tag_index += 1
    return aligned_synsets


def main():
    for dataset in ['train', 'valid', 'test']:
        dataset_name = dataset + '.' + LANG
        dataset_synsets_name = dataset_name + '_synsets'
        text = None
        with open(os.path.join(TOKENIZED_TEXT_FILES_PATH, dataset_name), 'r') as file:
            text = file.read()
        text_chunks = get_chunks(text, CHAR_LIMIT)
        read_synsets = None
        with open(os.path.join(TOKENIZED_TEXT_FILES_PATH, dataset_synsets_name), 'r') as file:
            read_synsets = file.read()
        parsed_chunks = literal_eval(read_synsets)
        index_aligned_chunks = align_indices(text_chunks, parsed_chunks)
        assigned_synsets = assign_synsets(synsets=flatten(index_aligned_chunks), text=text)
        POS_path = os.path.join(TOKENIZED_TEXT_FILES_PATH, dataset_name + '_postags')
        assigned_synsets = assign_POS_to_unknown_synsets(assigned_synsets, POS_path)
        with open(os.path.join(BPE_TEXT_FILES_PATH, dataset + '.bpe.' + LANG), 'r') as f:
            text_bpe = f.read()
        final_synsets = align_synsets_bpe(assigned_synsets, text_bpe)
        with open(os.path.join(BPE_TEXT_FILES_PATH, dataset + '.bpe.' + LANG + '_synsets_at'), 'w') as f:
            f.write(final_synsets)


if __name__ == "__main__":
    main()

