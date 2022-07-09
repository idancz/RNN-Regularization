from train_and_eval import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


global train_perplexity_dict
global validation_perplexity_dict
global test_perplexity_dict
global learning_rate_dict


def perplexity_plot(model_name):
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax2 = ax1.twiny()
    x1 = np.arange(len(train_perplexity_dict[model_name]))
    x2 = [f"{lr:.4f}" for lr in learning_rate_dict[model_name]]
    graph1 = ax1.plot(x1, train_perplexity_dict[model_name], label='Train')
    ax2.plot(x2, train_perplexity_dict[model_name], alpha=0)
    graph2 = ax1.plot(validation_perplexity_dict[model_name], label='Validation')
    graphs = graph1 + graph2
    labels = [g.get_label() for g in graphs]
    ax1.set_xticks(x1)
    ax2.set_xticks(x2)
    plt.title(model_name + " Perplexity Graph\n")
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Learning Rate")
    ax1.set_ylabel("Perplexity")
    ax1.legend(graphs, labels)
    ax1.grid()
    fig.set_size_inches((12, 8))
    plt.show()


def summary_table():
    best_val_ppl = {m: min(validation_perplexity_dict[m]) for m in validation_perplexity_dict}
    best_train_ppl = {m: min(train_perplexity_dict[m]) for m in train_perplexity_dict}
    best_test_ppl = {m: min(test_perplexity_dict[m]) for m in test_perplexity_dict}
    perplexity_dataframe = pd.DataFrame((best_train_ppl, best_val_ppl, best_test_ppl), index=[' Train Perplexity', 'Validation Perplexity', 'Test Perplexity'])
    perplexity_dataframe = perplexity_dataframe.T.round(3)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('expand_frame_repr', False)
    title = '##########> Penn Tree Bank - RNN - Summary Table - Best Perplexity <##########'
    print("______________________________________________________________________________")
    print(title)
    print("______________________________________________________________________________")
    print(perplexity_dataframe)