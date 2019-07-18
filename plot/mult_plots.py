import matplotlib.pyplot as plt

BASE_PATH = '/home/jordiae/Downloads/backup_logs/2/logs/'
PATHS = ['train15-bpe-synsets-pos-at-multiple.log', 'train16-bpe-synsets-pos-at-one-encoder.log',\
         'train18-bpe-synsets-pos-at-multiple-encoder_red.log', 'train17-bpe-synsets-pos-at-one-encoder_red.log', \
         'train20.log',  'train21-bpe-synsets-pos-at-one-encoder-sum.log', \
         'train22-one-encoder-synsets-subword-tags.log','train23-one-encoder-synsets-subword-tags-freq.log', \
         'train24-one-encoder-synsets-subword-tags-red256.log'
         ]
PATHS = list(map(lambda x: BASE_PATH + x, PATHS))
NAMES = ['Multi-encoder / 512+512 /\n concat / BPE tag',
         'Single-encoder / 512+512 /\n  concat / BPE tag',
         'Multi-encoder / 512+128 /\n  concat / BPE tag',
         'Single-encoder / 512+128 /\n  concat / BPE tag',
         'Multi-encoder / 512+512 /\n  sum / BPE tag',
         'Single-encoder / 512+512 /\n  sum / BPE tag',
         'Single-encoder / 512+128+sub(4) /\n  concat / NO BPE tag',
         'Single-encoder / 512+128+sub(4) /\n sum / Freq. thres.',
         'Single-encoder / 512+256+sub(4) /\n  concat / NO BPE tag',
         ]


def get_losses(log,path):
    epochs = []
    for l in log.splitlines()[1:]:
        try:
            if l.split()[1] != 'epoch':
                continue
        except:
            continue
        for index, token in enumerate(l.split()):
            if token == 'loss':
                loss = l.split()[index+1]
            elif token == 'valid_loss':
                valid_loss = l.split()[index+1]
                epochs.append({'loss': loss, 'valid_loss': valid_loss})
                break
    return epochs


def plot(paths, names):
    MEDIUM_SIZE = 30
    B_SIZE = 50

    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=B_SIZE)
    plt.rc('figure', titlesize=B_SIZE)
    N = len(paths)/3
    fig = plt.figure(figsize=(N*10, N*10))
    fig.subplots_adjust(hspace=0.5)
    #plt.tight_layout()
    first = True
    for index, path in enumerate(paths):
        i = index % N
        j = index // N
        l = get_losses(open(path, 'r').read(),path)
        train = [None]
        valid = [None]
        for x in l:
            train.append(float(x['loss']))
            valid.append(float(x['valid_loss']))
        fig.add_subplot(N, N, i * N + j + 1)
        if first:
            plt.plot(list(range(0, len(train))),train,label='train')
            plt.plot(list(range(0, len(train))),valid,label='valid')
            first = False
        else:
            plt.plot(list(range(0, len(train))), train)
            plt.plot(list(range(0, len(train))), valid)
        plt.xticks(list(range(5, 50, 5)))
        plt.yticks(list(range(0, 11, 1)))
        plt.xlim(0)
        #plt.legend()
        plt.xlabel(xlabel='Epoch')
        plt.ylabel(ylabel='Cross-entropy loss')
        plt.title(names[index])
    plt.figlegend()
    fig.suptitle('Factored architectures training with synsets')
    plt.savefig('plots/trainsynsets.png')
    #plt.show()


def main():
    plot(PATHS, NAMES)


if __name__ == "__main__":
    main()
