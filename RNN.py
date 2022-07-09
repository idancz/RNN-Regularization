import torch
import torch.nn as nn
from torch.autograd import Variable

# Need to turn-on GPU runtime for faster training, using CUDA.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1111)


class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.3):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp) # Token2Embeddings
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(ninp, nhid, nlayers) #(seq_len, batch_size, emb_size)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(ninp, nhid, nlayers) #(seq_len, batch_size, emb_size)
        else:
            print("ERROR!!\nAn invalid rnn_type option!!")
            exit()
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        # input size(bptt, bsz)
        emb = self.drop(self.encoder(inputs))
        # emb size(bptt, bsz, embsize)
        # hid size(layers, bsz, nhid)
        output, hidden = self.rnn(emb, hidden)
        # output size(bptt, bsz, nhid)
        output = self.drop(output)
        # decoder: nhid -> ntoken
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return Variable(weight.new_zeros(self.nlayers, bsz, self.nhid)).to(device), Variable(weight.new_zeros(self.nlayers, bsz, self.nhid)).to(device) #LSTM
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()).to(device) #GRU