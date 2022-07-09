from torch.autograd import Variable
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class DataHandler(object):
    def __init__(self, _path):
        self.dictionary = Dictionary()
        # three tensors of word index
        self.train_data = self.tokenize(_path+'ptb.train.txt')
        self.val_data = self.tokenize(_path+'ptb.valid.txt')
        self.test_data = self.tokenize(_path+'ptb.test.txt')

    def tokenize(self, file_name):
        # assert os.path.exists(path)
        # Add words to the dictionary
        with open(file_name, 'r') as f:
            tokens = 0
            for line in f:
                # line to list of token + eos
                words = ['<sos>'] + line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(file_name, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def batchify(self, data_type, bsz):
        if data_type == 'train':
            data = self.train_data
        elif data_type == 'test':
            data = self.test_data
        elif data_type == 'validation':
            data = self.val_data
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    def get_batch(self, data_set, i, seq_length):
        seq_len = min(seq_length, len(data_set) - 1 - i)
        inputs = Variable(data_set[i:i+seq_len].clone().detach())
        target = Variable(data_set[i+1:i+1+seq_len].clone().detach().view(-1))
        return inputs, target
