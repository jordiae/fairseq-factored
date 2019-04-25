# Based in https://github.com/NadaBen/babelfy/blob/master/babelfy.ipynb
import urllib
import urllib.parse
import urllib.request
import json
import gzip

from io import BytesIO

import os

import datetime

from ast import literal_eval

import re

import itertools

TOKENIZED_TEXT_FILES_PATH = os.path.join('..', '..', '..', '..', 'data', 'iwslt14.tokenized.de-en', 'tmp')
BPE_TEXT_FILES_PATH = os.path.join('..','..','..','..','data','iwslt14-preprocessed-joined')
# PREPROCESSED_TEXT_FILES_PATH = os.path.join('..', '..', '..', '..', 'data', 'iwslt14-preprocessed-joined')
LANG = 'de'
LANG_BABEL = LANG.upper()
CHAR_LIMIT = 3500
SERVICE_URL = 'https://babelfy.io/v1/disambiguate'
KEY = 'KEY'
CANDS = 'TOP'
#  TH = '.01'
MATCH = 'EXACT_MATCHING'


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


def write_synsets_chunks(chunks, restore, file_path, dataset_name, keep_trying=True, flush_log=True, limit=999):
    chunks = chunks[restore:]
    with open(file_path, 'a') as file:
        if restore == 0:
            file.write('[')
        if restore == len(chunks) - 1:
            print('Already written until', restore, flush=flush_log)
            return
        index = None
        for index, chunk in enumerate(chunks):
            if index == limit:
                print('LIMIT?', flush=flush_log)
                print('Last chunk not written! Next time restore should be set to', index, flush=flush_log)
                print('Just in case, here you are! Last chunk NOT processed:', flush=flush_log)
                print(chunk, flush=flush_log)
                exit()
            if index < restore:
                continue
            print('chunk', index+1, 'of', len(chunks), 'at', dataset_name, flush=flush_log)
            synsets = get_synsets(chunk, flush_log, verbose=True)
            if synsets is None:
                if keep_trying:
                    print('Error! Re-trying...', flush=flush_log)
                    currentDT = datetime.datetime.now()
                    print('Started re-trying at', str(currentDT), flush=flush_log)
                    while synsets is not None:
                        synsets = get_synsets(chunk, flush_log, verbose=False)
                else:
                    print('Last chunk not written! Next time restore should be set to', index, flush=flush_log)
                    print('Just in case, here you are! Last chunk NOT processed:', flush=flush_log)
                    print(chunk, flush=flush_log)
                    exit()
            else:
                file.write(str(synsets))
                file.write(',')
        if index == len(chunks) - 1:
            file.write(']')

def get_synsets(chunk, flush_log, verbose):
    params = {
        'text': chunk,
        'lang': LANG_BABEL,
        'key': KEY,
        'cands': CANDS,
        'match': MATCH
        # 'th': th
    }

    url = SERVICE_URL + '?' + urllib.parse.urlencode(params)
    request = urllib.request.Request(url)
    try:
        request.add_header('Accept-encoding', 'gzip')
    except Exception as exception:
        if verbose:
            print(exception, flush=flush_log)
        return None
    response = urllib.request.urlopen(request)
    try:
        if response.info().get('Content-Encoding') == 'gzip':
            buf = BytesIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            data = json.loads(f.read().decode('utf-8'))
            synsets = []
            for result in data:
                # retrieving token fragment
                tokenFragment = result.get('tokenFragment')
                tfStart = tokenFragment.get('start')
                tfEnd = tokenFragment.get('end')
                # print (str(tfStart) + "\t" + str(tfEnd))

                # retrieving char fragment
                charFragment = result.get('charFragment')
                cfStart = charFragment.get('start')
                cfEnd = charFragment.get('end')
                # print (str(cfStart) + "\t" + str(cfEnd))

                # retrieving BabelSynset ID
                synsetId = result.get('babelSynsetID')
                # print(synsetId, tfStart, tfEnd)

                source = str(result.get('source'))
                DBpediaURL = str(result.get('DBpediaURL'))
                BabelNetURL = str(result.get('BabelNetURL'))
                globalScore =  str(result.get('globalScore'))
                coherenceScore = str(result.get('coherenceScore'))
                '''
                print ('source'+ "\t" +result.get('source'))
                print('DBpediaURL'+ "\t" +result.get('DBpediaURL'))
                print('BabelNetURL'+ "\t" +result.get('BabelNetURL'))
                print('globalScore'+ "\t" +str(result.get('globalScore')))
                print('coherenceScore'+ "\t" +str(result.get('coherenceScore')))
                
                '''

                '''
                new_syn = {'synsetId': synsetId, 'cfStart': cfStart, 'cfEnd': cfEnd, 'source': source,
                           'globalScore': globalScore, 'coherenceScore': coherenceScore, 'DBpediaURL': DBpediaURL,
                           'BabelNetURL': BabelNetURL, 'tfStart': tfStart, 'tfEnd': tfEnd}
                '''
                # new_syn = {'synsetId': synsetId, 'cfStart': cfStart, 'cfEnd': cfEnd}
                '''
                for element in new_syn:
                    if new_syn[element] is None:
                        if verbose:
                            print(element, 'is None!', flush=flush_log)
                        return None
                '''
                new_syn = [synsetId, cfStart, cfEnd]
                empty = False
                if synsetId is None or synsetId == '':
                    if verbose:
                        print('synsetId is None or empty string!', flush=flush_log)
                    empty = True
                if cfStart is None or cfStart == '':
                    if verbose:
                        print('cfStart is None or empty string!', flush=flush_log)
                    empty = True
                if cfEnd is None or cfEnd == '':
                    if verbose:
                        print('cfEnd is None or empty string!', flush=flush_log)
                    empty = True
                if empty:
                    return None

                synsets.append(new_syn)
            return synsets
        else:
            if verbose:
                print('Not GZIP!', flush=flush_log)
            return None
    except Exception as exception:
        if verbose:
            print('failed at reading the response!', flush=flush_log)
            print(exception, flush=flush_log)
        return None


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
    for synset, start_synset, end_synset in synsets:
        end_synset += 1  # Babelfy synsets have end of word as subset, unlike Python slices/indices.
        word_index = index_dict[start_synset]
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
        else:
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
                    max = synset_dict
                    chosen_synset = synset
            assigned_synsets.append(chosen_synset)
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
