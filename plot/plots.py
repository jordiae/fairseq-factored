import sys
def get_losses(log):
    epochs = []
    for l in log.splitlines()[1:]:
        if l.split()[1] != 'epoch':
            continue
        for index, token in enumerate(l.split()):
            if token == 'loss':
                loss = l.split()[index+1]
            elif token == 'valid_loss':
                valid_loss = l.split()[index+1]
                epochs.append({'loss': loss, 'valid_loss': valid_loss})
                break
    return epochs

import matplotlib.pyplot as plt
PATH = '/home/jordiae/Downloads/backup_logs/2/logs/train14-bpe-pos-at-al-one-encoder-new.log'
def main():
    #logs = get_logs(sys.argv[1:])
    #l = get_losses(log)
    l = get_losses(open(PATH, 'r').read())
    train = [None]
    valid = [None]
    for x in l:
        train.append(float(x['loss']))
        valid.append(float(x['valid_loss']))
    print(l)
    print(len(l))
    x, y = zip(*l)  # unpack a list of pairs into two tuples
    print(x)
    print(train)
    print(valid)
    print(len(train), len(valid))
    plt.plot(list(range(0, len(train))),train,label='train')
    plt.plot(list(range(0, len(train))),valid,label='valid')
    plt.xticks(list(range(5, 50, 5)))
    plt.yticks(list(range(0, 11, 1)))
    plt.xlim(0)
    plt.legend()
    plt.xlabel(xlabel='Epoch')
    plt.ylabel(ylabel='Cross-entropy loss')
    plt.title('Single-encoder with concatenation and PoS (512+32)')
    plt.savefig('plots/train14pos.png')
    #plt.show()

if __name__ == "__main__":
    main()