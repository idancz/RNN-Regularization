from graph_and_stats import *
import pickle

global train_perplexity_dict
global validation_perplexity_dict
global test_perplexity_dict
global learning_rate_dict
global path, data_path, models_path


def main():
    print(device)
    ########################################################################################################
    ################################################# Train ################################################
    ########################################################################################################
    # models -- > "LSTM", "GRU"
    # optimizers --> "SGD", "SGD_Momentum", "SGD_WeightDecay", "Adam"  (when using Adam make sure LearningRate < 0.01)
    ################################################# LSTM #################################################
    ModelType = "LSTM"
    OptType = "SGD"
    LearningRate = 10
    Epochs = 15
    BatchSize = 20
    SequenceLength = 20
    Dropout = 0


    trained, name = build_and_train(False, ModelType, OptType, Dropout, LearningRate, Epochs, BatchSize, SequenceLength)

    test_loss, test_perplexity = trained.evaluate('test')
    test_perplexity_dict[name] = [test_perplexity]
    print('-' * 127)
    print(f'{name} | test loss: {test_loss:.3f} | test perplexity {test_perplexity:.3f}')
    print('-' * 127)

    perplexity_plot(name)
    ########################################## LSTM_With_Dropout ###########################################
    ModelType = "LSTM"
    OptType = "SGD"
    LearningRate = 10
    Epochs = 30
    BatchSize = 20
    SequenceLength = 20
    Dropout = 0.5


    trained, name = build_and_train(False, ModelType, OptType, Dropout, LearningRate, Epochs, BatchSize, SequenceLength)

    test_loss, test_perplexity = trained.evaluate('test')
    test_perplexity_dict[name] = [test_perplexity]
    print('-' * 127)
    print(f'{name} | test loss: {test_loss:.3f} | test perplexity {test_perplexity:.3f}')
    print('-' * 127)

    perplexity_plot(name)

    ################################################# GRU #################################################
    ModelType = "GRU"
    OptType = "SGD_WeightDecay"
    LearningRate = 5
    Epochs = 25
    BatchSize = 20
    SequenceLength = 20
    Dropout = 0


    trained, name = build_and_train(False, ModelType, OptType, Dropout, LearningRate, Epochs, BatchSize, SequenceLength)

    test_loss, test_perplexity = trained.evaluate('test')
    test_perplexity_dict[name] = [test_perplexity]
    print('-' * 127)
    print(f'{name} | test loss: {test_loss:.3f} | test perplexity {test_perplexity:.3f}')
    print('-' * 127)

    perplexity_plot(name)

    ########################################## GRU_With_Dropout ############################################
    ModelType = "GRU"
    OptType = "SGD"
    LearningRate = 5
    Epochs = 30
    BatchSize = 20
    SequenceLength = 20
    Dropout = 0.5


    trained, name = build_and_train(False, ModelType, OptType, Dropout, LearningRate, Epochs, BatchSize, SequenceLength)

    test_loss, test_perplexity = trained.evaluate('test')
    test_perplexity_dict[name] = [test_perplexity]
    print('-' * 127)
    print(f'{name} | test loss: {test_loss:.3f} | test perplexity {test_perplexity:.3f}')
    print('-' * 127)

    perplexity_plot(name)
    #######################################################################################################

    summary_table()

    # with open(path+'train_perplexity_dict.pickle', 'wb') as handle:
    #     pickle.dump(train_perplexity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(path+'validation_perplexity_dict.pickle', 'wb') as handle:
    #     pickle.dump(validation_perplexity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(path+'test_perplexity_dict.pickle', 'wb') as handle:
    #     pickle.dump(test_perplexity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(path+'learning_rate_dict.pickle', 'wb') as handle:
    #     pickle.dump(learning_rate_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




    # with open(path+'validation_perplexity_dict.pickle', 'rb') as handle:
    #     validation_perplexity_dict = pickle.load(handle)
    #
    # with open(path+'train_perplexity_dict.pickle', 'rb') as handle:
    #     train_perplexity_dict = pickle.load(handle)

    # with open(path+'testt_perplexity_dict.pickle', 'rb') as handle:
    #     test_perplexity_dict = pickle.load(handle)

    # with open(path+'learning_rate_dict.pickle', 'rb') as handle:
    #     learning_rate_dict = pickle.load(handle)
    #######################################################################################################
    ################################################# Test ################################################
    #######################################################################################################
    ModelName = "LSTM"
    pre_trained_model = load_model(ModelName)
    print(pre_trained_model)
    trained_test_loss, trained_model_perplexity = test_model(pre_trained_model)
    print(f"\nThe {ModelName} pre trained model test loss is: {trained_test_loss:.3f} and test perplexity is: {trained_model_perplexity:.3f}\n")
    #####################################################################################################################################

    ModelName = "LSTM_With_Dropout_0.5"
    pre_trained_model = load_model(ModelName)
    print(pre_trained_model)
    trained_test_loss, trained_model_perplexity = test_model(pre_trained_model)
    print(f"\nThe {ModelName} pre trained model test loss is: {trained_test_loss:.3f} and test perplexity is: {trained_model_perplexity:.3f}\n")
    #####################################################################################################################################

    ModelName = "GRU"
    pre_trained_model = load_model(ModelName)
    print(pre_trained_model)
    trained_test_loss, trained_model_perplexity = test_model(pre_trained_model)
    print(f"\nThe {ModelName} pre trained model test loss is: {trained_test_loss:.3f} and test perplexity is: {trained_model_perplexity:.3f}\n")
    #####################################################################################################################################

    ModelName = "GRU_With_Dropout_0.5"
    pre_trained_model = load_model(ModelName)
    print(pre_trained_model)
    trained_test_loss, trained_model_perplexity = test_model(pre_trained_model)
    print(f"\nThe {ModelName} pre trained model test loss is: {trained_test_loss:.3f} and test perplexity is: {trained_model_perplexity:.3f}\n")
    #####################################################################################################################################


if __name__ == '__main__':
    main()
