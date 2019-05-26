import os
from ast import literal_eval
import re

import sentencepiece as spm

FLORES_DATA_PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/flores'
BPE_MODEL_PATH = os.path.join(FLORES_DATA_PATH, 'data-bin', 'wiki_ne_en_bpe5000', 'sentencepiece.bpe.model')
s = spm.SentencePieceProcessor()
s.Load(BPE_MODEL_PATH)

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data/wiki_ne_en_bpe5000'
LANG = 'en'
CHAR_LIMIT = 3500


def get_chunks(s, n_chars=CHAR_LIMIT):
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

def normalize_token(token):
    pieces = s.EncodeAsPieces(token)
    return ''.join(pieces).replace('\u2581', '')


def align_sentencepiece(text_bpe, text_token, text_synset):
    i = 0
    print(len(text_bpe.splitlines()), len(text_token.splitlines()), len(text_synset.splitlines()))
    repeated_synsets = ''
    for line_bpe, line_token, line_synset in zip(text_bpe.splitlines(), text_token.splitlines(), text_synset.splitlines()):
        index_bpe = 0
        i += 1
        # print(len(line_bpe.split()),len(line_token.split()), len(line_synset.split()))
        if i - 1 % 1000 == 0:
            print("Aligned", i, "of", len(text_bpe.splitlines()), flush=True)
        for index, token in enumerate(line_token.split()):
            token = normalize_token(token)
            current_word = ''
            counter = 0
            currently_in_space = False
            while not current_word == token:
                if index_bpe >= len(line_bpe):
                    print(token)
                    print('current_word', current_word)
                    print('out of bounds', line_token, index_bpe, len(line_bpe)-1)
                    exit()
                if line_bpe[index_bpe] == '\u2581' or line_bpe[index_bpe] == ' ':
                    index_bpe += 1
                    if not currently_in_space:
                        counter += 1
                    currently_in_space = True
                else:
                    current_word += line_bpe[index_bpe]
                    index_bpe += 1
                    currently_in_space = False
            if counter == 1:
                repeated_synsets += line_synset[index] + ' '
            else:
                repeated_synsets += line_synset[index] + ' '
                counter -= 1
                while counter > 1:
                    repeated_synsets += line_synset[index] + ' '
                    counter -= 1
                    repeated_synsets += line_synset[index] + ' '
            current_word = ''
        '''
        try:
            repeated_synsets = repeated_synsets[:-1] + '\n'
        except:
            repeated_synsets += '\n'
        '''
        repeated_synsets += '\n'
    return repeated_synsets


def flatten(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


def assign(synsets, original_text):
    chars_text = [set()] * len(original_text)
    for synset, start_synset, end_synset in synsets:
        end_synset += 1
        for i in range(start_synset, end_synset):
            chars_text[i].add((synset, start_synset, end_synset))
    cache = {}
    for index in range(0, len(original_text)):
        if index % 100 == 0: print('char',index+1,'of',len(original_text),flush=True)
        if chars_text[index] in cache:
            chars_text[index] = cache[chars_text[index]]
        else:
            if len(chars_text[index]) == 0:
                chars_text[index] = None
                selected_synset = None
            elif len(chars_text[index]) == 1:
                chars_text = list(chars_text[index])[0]
                selected_synset = list(chars_text[index])[0]
            else:
                max_len = 0
                selected_synset = None
                for synset, start_synset, end_synset in chars_text[index]:
                    if end_synset-start_synset > max_len:
                        max_len = end_synset-start_synset
                        selected_synset = synset
                chars_text[index] = selected_synset
            cache[chars_text[index]] = selected_synset
    return chars_text



def chars_to_tokens3(chars_assigned_synsets, text_token, text_pos):
    tokenized_synsets = ''
    for token_line, pos_line in zip(text_token.splitlines(), text_pos.splitlines()):
        for (index, (token, (start_token, end_token))) in enumerate(zip(token_line.split(), [(m.start(), m.end()) for m in re.finditer(r'\S+', token_line)])):
            if chars_assigned_synsets[start_token] is None:
                tokenized_synsets += pos_line[index] # 'NONE'
            else:
                tokenized_synsets += chars_assigned_synsets[start_token]
            #tokenized_synsets += ' '
            if index == len(token_line.split()):
                tokenized_synsets += '\n'
            else:
                tokenized_synsets += ' '
        #tokenized_synsets = tokenized_synsets[:-1] + '\n'
        # print('Added 1 line...')
    return tokenized_synsets

def main():
    for dataset in ['train', 'valid', 'test']:
        print('Loading', dataset, flush=True)
        with open(os.path.join(PATH, dataset + '.' + LANG), 'r') as file:
            text = file.read()
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG), 'r') as file:
            text_bpe = file.read()
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tokens'), 'r') as file:
            text_token = file.read()
        '''
        with open(os.path.join(PATH, dataset + '.' + LANG + '_lemmas'), 'r') as file:
            text_lemma = file.read()
        '''
        with open(os.path.join(PATH, dataset + '.' + LANG + '_pos'), 'r') as file:
            text_pos = file.read()
        '''
        with open(os.path.join(PATH, dataset + '.' + LANG + '_deps'), 'r') as file:
            text_dep = file.read()
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tags'), 'r') as file:
            text_tag = file.read()
        '''
        with open(os.path.join(PATH, dataset + '.' + LANG + '_synsets'), 'r') as file:
            read_synsets = file.read()
        print('Getting chunks',flush=True)
        text_chunks = get_chunks(text)
        print('Parsing chunks',flush=True)
        parsed_chunks = literal_eval(read_synsets)
        print('Aligning indices',flush=True)
        index_aligned_chunks = align_indices(text_chunks, parsed_chunks)
        print('Assigning synsets',flush=True)
        chars_assigned_synsets = assign(synsets=flatten(index_aligned_chunks), original_text=text)
        #print('Assigning POS if unknown',flush=True)
        #chars_assigned_synsets = assign_pos_if_unknown_synset(chars_assigned_synsets, text_token, text_pos)
        print('Tokenizing chars and assigning POS if unknown synset', flush=True)
        tokenized_synsets = chars_to_tokens3(chars_assigned_synsets, text_token, text_pos)
        #print('Assigning POS if unknown',flush=True)
        print(len(tokenized_synsets.splitlines()))
        print('Aligning sentencepiece', flush=True)
        repeated_synsets = align_sentencepiece(text_bpe, text_token, tokenized_synsets)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_synsets'), 'w') as file:
            file.write(repeated_synsets)


if __name__ == "__main__":
    main()
