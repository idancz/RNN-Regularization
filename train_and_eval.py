from torch import optim
from data_handeling import *
from RNN import *
import numpy as np
import math
import time


# Trainer Class which gets training parameters :
train_perplexity_dict = {}
validation_perplexity_dict = {}
test_perplexity_dict = {}
learning_rate_dict = {}
path = ""
data_path = "data/"
models_path = "models/"


class Trainer:
    def __init__(self, model, optimizer, criterion, lr, epochs, batch_size, seq_length, data_handler):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.data_handler = data_handler

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if self.model.rnn_type == "LSTM":
            return tuple([Variable(each.clone().detach()) for each in h])
        return Variable(h.data)

    # train function of class Trainer.
    def train(self):
        self.model.train()
        total_loss = 0
        num_of_tokens = len(self.data_handler.dictionary)
        hidden = self.model.init_hidden(self.batch_size)
        trained_data_batches = self.data_handler.batchify('train', self.batch_size)
        for batch_idx, i in enumerate(range(0, trained_data_batches.size(0) - 1, self.seq_length)):
            inputs, targets = self.data_handler.get_batch(trained_data_batches, i, self.seq_length)
            inputs, targets = inputs.to(device), targets.to(device)

            # Wraps hidden states in new Variables, to detach them from their history.
            # Otherwise backprop through the entire training history.
            hidden = self.repackage_hidden(hidden)

            output, hidden = self.model(inputs, hidden)
            loss = self.criterion(output, targets)
            # loss = self.criterion(output.view(-1, num_of_tokens), targets)
            self.optimizer.zero_grad()
            loss.backward()
            # helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)  # clip=1.5 for best results
            self.optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            perplexity = math.exp(avg_loss)
        return tuple([avg_loss, perplexity])

    # test function for class testing
    def evaluate(self, data_type):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            eval_batch_size = 10
            # num_of_tokens = len(self.data_handler.dictionary)
            hidden = self.model.init_hidden(eval_batch_size)  # self.batch_size
            evaluate_data_batches = self.data_handler.batchify(data_type, eval_batch_size)
            for batch_idx, i in enumerate(range(0, evaluate_data_batches.size(0) - 1, self.seq_length)):
                inputs, targets = self.data_handler.get_batch(evaluate_data_batches, i, self.seq_length)
                inputs, targets = inputs.to(device), targets.to(device)

                output, hidden = self.model(inputs, hidden)
                loss = self.criterion(output, targets)

                # total_loss += loss.item()
                total_loss += len(inputs) * loss.data
                hidden = self.repackage_hidden(hidden)

        avg_loss = total_loss / len(evaluate_data_batches)
        perplexity = math.exp(avg_loss)
        return tuple([avg_loss, perplexity])

    # run function for training and evaluation
    def run(self, model_name="LSTM"):
        global train_perplexity_dict
        global validation_perplexity_dict
        global learning_rate_dict
        global models_path
        divider = 4.0
        best_perplexity = np.inf
        learning_rate_dict[model_name] = []
        train_perplexity_dict[model_name] = []
        validation_perplexity_dict[model_name] = []
        print(f'{50*"*"} Starts running {model_name} {50*"*"}')
        for epoch in range(1, self.epochs+1):
            start_time = time.time()
            train_loss, train_perplexity = self.train()
            validation_loss, validation_perplexity = self.evaluate('validation')
            learning_rate_dict[model_name].append(self.lr)
            train_perplexity_dict[model_name].append(train_perplexity)
            validation_perplexity_dict[model_name].append(validation_perplexity)
            print(f'epoch {epoch} | time : {(time.time() - start_time):5.2f}s | lr: {self.lr:.4f} | train loss : {train_loss:.3f} | train perplexity: {train_perplexity:.3f} | validation loss: {validation_loss:.3f} | validation perplexity: {validation_perplexity:.3f}')
            if validation_perplexity < best_perplexity:
                torch.save(self.model, models_path + model_name + '.pt')
                best_perplexity = validation_perplexity
            else:
                if self.optimizer.__class__.__name__ == 'SGD':
                    self.lr /= divider
                    for group in self.optimizer.param_groups:
                        group['lr'] = self.lr


# main calling function or constructing and training
def build_and_train(pre_trained_cont=False, model_type="LSTM", opt_type="SGD", model_dropout=0, learning_rate=1, epochs=20, batch_size=20, seq_length=20):
    global data_path
    # Data Loader
    data_handler = DataHandler(data_path)
    if model_type not in ["LSTM", "GRU"]:
        print("ERROR!!")
        print(f'model name {model_type} not in models list: LSTM, GRU')
        exit()
    if opt_type not in ["SGD", "SGD_Momentum", "SGD_WeightDecay", "Adam"]:
        print("ERROR!!")
        print(f'optimizer type {opt_type} not in optimizers list: SGD, SGD_Momentum, SGD_WeightDecay, Adam')
        exit()

    if model_dropout > 0:
        model_name = model_type+"_With_Dropout_"+str(model_dropout)
    else:
        model_name = model_type

    models = {
        "LSTM": lambda: RNNModel("LSTM", len(data_handler.dictionary), ninp=200, nhid=200, nlayers=2, dropout=model_dropout),
        "GRU": lambda: RNNModel("GRU", len(data_handler.dictionary), ninp=200, nhid=200, nlayers=2, dropout=model_dropout),
    }

    if pre_trained_cont:
        model = torch.load(models_path+model_name+".pt").to(device)
    else:
        model = models[model_type]().to(device)
    print(model)

    # Load loss function
    criterion = nn.CrossEntropyLoss()

    optimizers = {
        "Adam": lambda: optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.99)),
        "SGD": lambda: optim.SGD(model.parameters(), lr=learning_rate),
        "SGD_Momentum": lambda: torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8),
        "SGD_WeightDecay": lambda: torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    }

    # Build optimizer
    optimizer = optimizers[opt_type]()
    print(optimizer)

    # Create Trainer with all the parameters
    trainer = Trainer(model, optimizer, criterion, learning_rate, epochs, batch_size, seq_length, data_handler) # model, optimizer, criterion, lr, epochs, batch_size, seq_length, data_handler
    trainer.run(model_name)
    return trainer, model_name


# loading pre trained model , getting model_name as parameter
def load_model(model_name):
    global models_path
    print(f'Loading {models_path+model_name+".pt"}')
    model = torch.load(models_path+model_name+".pt").to(device)
    model.eval()  # evaluating only
    return model


# testing pre-trained model
def test_model(model):
    global data_path
    data_handler = DataHandler(data_path)
    trainer = Trainer(model=model, optimizer=None,
                  criterion=nn.CrossEntropyLoss(),
                  lr=None,
                  epochs=1,
                  batch_size=10,
                  seq_length=20,
                  data_handler=data_handler)
    _test_loss, _test_perplexity = trainer.evaluate('test')
    return _test_loss, _test_perplexity