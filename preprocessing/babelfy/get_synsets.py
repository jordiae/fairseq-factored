# Based in https://github.com/NadaBen/babelfy/blob/master/babelfy.ipynb
import urllib
import urllib.parse
import urllib.request
import json
import gzip

from io import BytesIO

import os

import datetime


TOKENIZED_TEXT_FILES_PATH = os.path.join('..', '..', '..', '..', 'data', 'iwslt14.tokenized.de-en', 'tmp')
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
            current_chunk = ''
            current_char_count = 0
    if len(current_chunk) > 0:
        chunks.append(current_chunk)
    return chunks


def write_synsets_chunks(chunks, restore, file_path, dataset_name, keep_trying = True):
    chunks = chunks[restore:]
    with open(file_path, 'w') as file:
        if restore == 0:
            file.write('[')
        if restore == len(chunks) - 1:
            print('Already written until', restore)
            return
        for index, chunk in enumerate(chunks):
            if index == 995:
                print('LIMIT?')
                exit()
            if index < restore:
                continue
            print('chunk', index+1, 'of', len(chunks), 'at', dataset_name)
            synsets = get_synsets(chunk)
            if synsets is None:
                if keep_trying:
                    print('Error! Re-trying...')
                    currentDT = datetime.datetime.now()
                    print('Started re-trying at', str(currentDT))
                    while synsets is not None:
                        synsets = get_synsets(chunk)
                else:
                    print('Last chunk not written! Restore should be set to', index, 'next time')
                    exit()
            else:
                file.write(str(synsets))
                file.write(',')
        if index == len(chunks) - 1:
            file.write(']')


def get_synsets(chunk):
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
        print(exception)
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
                new_syn = {'synsetId': synsetId, 'cfStart': cfStart, 'cfEnd': cfEnd, 'source': source,
                           'globalScore': globalScore, 'coherenceScore': coherenceScore, 'DBpediaURL': DBpediaURL,
                           'BabelNetURL': BabelNetURL, 'tfStart': tfStart, 'tfEnd': tfEnd}
                for element in new_syn:
                    if new_syn[element] is None:
                        print(element, 'is None!')
                        return None
                synsets.append(new_syn)
            return synsets
        else:
            print('Not GZIP!')
            return None
    except Exception as exception:
        print('failed at reading the response!')
        print(exception)
        return None


def main():
    for dataset in  ['train', 'valid', 'test']:
        dataset_name = dataset + '.' + LANG
        with open(os.path.join(TOKENIZED_TEXT_FILES_PATH, dataset_name), 'r') as file:
            text = file.read()
        chunks = get_chunks(text, CHAR_LIMIT)
        synsets_file_path = os.path.join(TOKENIZED_TEXT_FILES_PATH, dataset + '.' + LANG + '_synsets')
        write_synsets_chunks(chunks=chunks, restore=0, file_path=synsets_file_path, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
