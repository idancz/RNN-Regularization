# RNN-Regularization
Implementation of RNN (LSTM/GRU) on "Penn Tree Bank" dataset" based on "Recurrent Neural Network Regularization" paper ("small" model), by Zaremba.<br />
Link to paper --> https://arxiv.org/pdf/1409.2329.pdf

# Program Description
## Background
The main key features that I used are:
1. Adding ‘<sos>’ at the start of every sentence and ‘<eos>’ at the end of evert sentence during tokenize the data (DataHandler class).
2. Using Embedding layer to represent the input vocabulary tokens
3. Apply Dropout only on the input and output of the model without applying on the recurrent
4. Initialized the model’s weights uniformly between [-0.1, 0.1]
5. Apply clipping to prevent the exploding gradient problem in RNNs using clip=1.5
6. Tracking and modifying the learning rate (when using ‘SGD’) respectively to the validation perplexity improvement as follow:
   - If validation perplexity decrease from the previous epoch -- > learning rate /= 4

## Training
Train using the function build_and_train(pre_trained_cont, model_type, opt_type, model_dropout, learning_rate, epochs, batch_size, seq_length)<br />
1. pre_trained_cont – could be False to train new model/ True load and train pre trained model.
2. model_type – could be “LSTM” or “GRU”
3. opt_type – could be "SGD", "SGD_Momentum", "SGD_WeightDecay", "Adam"
4. model_dropout – should be between 0 to 1 (when 0 means no dropout)
5. learning_rate – any number
6. epochs – integer number
7. batch_size – integer number
8. seq_lenght – integer number
<br />
The function returns trained model and model_name (related to the saved model file).
<br />
I have prepared training blokes for each one of the models with defined inputs for the best results<br />
Please comment/uncomment relevant block for training.<br />
Example:<br />
![image](https://user-images.githubusercontent.com/108329249/178115922-3c27cbff-20c5-4b94-a331-de478afc5469.png)

## Load and Test
Loading pre trained model using the function load_model(model_name)
<br />
1. model_name – should gets the <saved_model_name>.pt
2. Testing model using the function test_model(model)
3. model – should be trained model
<br />
I have prepared loading and testing blokes for each one of the trained models<br />
Please comment/uncomment relevant block for testing.<br />
Example:<br />
![image](https://user-images.githubusercontent.com/108329249/178115928-1e195ff6-b9e4-4d50-9454-1773cbb11ba4.png)

## Results
I found that using Dropout has shown best performance with lower test perplexity for both LSTM and GRU,<br />
the Dropout layers helps to reduce overfitting LSTM with Dropout gave the best result overall.<br />
While comparing the models without Dropout, the GRU gave better results than LSTM when I used SGD with weight_decay=0.0001.<br />

You can see below summary table of our results:<br />
![image](https://user-images.githubusercontent.com/108329249/178115110-5fd98ba5-ef90-4f88-9605-3ed1b8f8cffc.png)

## Graphs
### LSTM using SGD optimizer  test perplexity = 91.25
![image](https://user-images.githubusercontent.com/108329249/178115513-7b92e32f-fd08-4a5a-b6ca-9a2e3011be76.png)

### LSTM with Dropout=0.5 model using SGD optimizer  test perplexity = 76.569


### GRU using SGD with Weight_Decay=0.0001 optimizer  test perplexity = 90.968


### GRU with Dropout=0.5 using SGD optimizer  test perplexity = 80.41



