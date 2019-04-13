#import treetaggerwrapper
import os
import sys
import itertools

def get_pos_tagger(path, pos_tagger_link, links):
    if not os.path.exists(path):
        os.system('wget ' + pos_tagger_link)
        os.system('tar -xvf '+ path +'.tar.gz ' + '--one-top-level=' + path)
        for link in links:
            os.system('wget ' + link + ' -P ' + path)
        os.chdir(path)
        os.system('tar -xvf tagger-scripts.tar.gz')
       
        os.system('gzip -d german.par.gz')
        os.system('gzip -d german-chunker.par.gz')
        os.system('sh install-tagger.sh')
        os.chdir('..')
        os.system('mv utf8-tokenize.perl ' + os.path.join(path,'cmd','utf8-tokenize.perl'))

#def apply_pos_tagger(tagger,text):
#    pos_tags_list = list(map(lambda tag: tag.pos,treetaggerwrapper.make_tags(tagger.tag_text(text))))
#    pos_tags_str = ' '.join(str(x) for x in pos_tags_list)
#    return pos_tags_str

def apply_pos_tagger(text_file, output_tags_file, tagger_path):
    os.system("cat " + text_file + " | sed -E 's/(([a-z])|[0-9])--/\1/g' | sed -E 's/\.\.\.\.+/\.\.\./g' | " + tagger_path + "/cmd/tree-tagger-german | awk '{ print $2 }' | tr -s '\n' '\n' > " + output_tags_file)

def align_pos_tags_bpe(text_bpe,tags):
    token_index = 0
    tag_index = 0
    splitted_text_bpe = text_bpe.split()
    splitted_tags = tags.split()
    aligned_tags = ''
    end_of_lines_positions = list(itertools.accumulate(list(map(lambda x: len(x.split()),text_bpe.splitlines()))))
    end_of_line_index = 0
    while token_index < len(splitted_text_bpe):
        if '@@' in splitted_text_bpe[token_index]:
            aligned_tags += '@@'
            while '@@' in splitted_text_bpe[token_index]:
                #print(splitted_text_bpe[token_index],splitted_tags[tag_index])
                aligned_tags += (splitted_tags[tag_index])
                aligned_tags += ' '
                token_index += 1
                if '@@' not in splitted_text_bpe[token_index]:
                    #print(splitted_text_bpe[token_index],splitted_tags[tag_index])
                    aligned_tags += (splitted_tags[tag_index])
                    if token_index == end_of_lines_positions[end_of_line_index] - 1:
                        aligned_tags += '\n'
                        end_of_line_index += 1
                    else:
                        aligned_tags += ' '
                    token_index += 1
                    tag_index += 1
        else:
            #print(splitted_text_bpe[token_index],splitted_tags[tag_index])
            aligned_tags += (splitted_tags[tag_index])
            if token_index == end_of_lines_positions[end_of_line_index] - 1:
                aligned_tags += '\n'
                end_of_line_index += 1
            else:
                aligned_tags += ' '
            token_index += 1
            tag_index += 1
    return aligned_tags

def main():
    POS_TAGGER_PATH = 'tree-tagger-linux-3.2.2'
    POS_TAGGER_LINK = 'http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.2.tar.gz'
    LINKS = ['http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz','http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh','http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german.par.gz', \
        'http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german-chunker.par.gz']
    get_pos_tagger(POS_TAGGER_PATH, POS_TAGGER_LINK, LINKS)   
    '''
    ENV_VAR = 'TAGDIR'
    os.environ[ENV_VAR] = POS_TAGGER_PATH
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='de')
    '''
    #text = 'Das ist ein Test.'
    TOKENIZED_TEXT_FILES_PATH = os.path.join('..','..','..','..','data','iwslt14.tokenized.de-en', 'tmp')
    BPE_TEXT_FILES_PATH = os.path.join('..','..','..','..','data','iwslt14-preprocessed-joined')
    LANG = 'de'
    for dataset in  ['train','valid','test']:
        apply_pos_tagger(os.path.join(TOKENIZED_TEXT_FILES_PATH, dataset + '.' + LANG), os.path.join(TOKENIZED_TEXT_FILES_PATH, dataset + '.' + LANG + '_postags'),POS_TAGGER_PATH)
        text_bpe = ''
        with open(os.path.join(BPE_TEXT_FILES_PATH, dataset + '.bpe.' + LANG),'r') as f:
            text_bpe = f.read()
        tags = ''
        with open(os.path.join(TOKENIZED_TEXT_FILES_PATH, dataset + '.' + LANG + '_postags_at'),'r') as f:
            tags = f.read()
        aligned_tags = align_pos_tags_bpe(text_bpe, tags)
        with open(os.path.join(BPE_TEXT_FILES_PATH, dataset + '.bpe.' + LANG + '_postags_at'),'w') as f:
            f.write(aligned_tags)
    '''
    with open(sys.argv[1],'r') as f:
        text = f.read()
    tags = apply_pos_tagger(tagger,text)
    print(tags)
    '''


if __name__ == "__main__":
    main()
