# DATA 4620 Homework 4: Sentiment Analysis

Sentiment analysis is the task of determining sentiment (e.g., positive or negative) from text.  In this homework you will experiment with sentiment analysis on a dataset of reviews from IMDB which have been annotated as having either positive or negative sentiment.  

The provided starter notebook shows how to download and unpack the data, and how to set up and use the BPE tokenizer from BERT.

You will compare a 1D CNN model with a GRU model for sentiment classification.  Each model should start with an Embedding layer to map the BPE tokens to embedding vectors.

Here are the common hyperparameters for the models:
- Embedding size: 100
- Maximum sequence length: 100
- Hidden layer size: 100
- Num hidden layers: 3
- 10 epochs, Adam optimizer with learning rate 3e-4


### Instructions
1. **Dataset preprocessing.**  Create a 90/10 train/val split of the data.  Lab 8.2 gives an example of a TokenDataset class which you can adapt for this assignment.  For the training set, use a maximum sequence length: if the text is too long, take a random sub-sequence; if the text is too short, pad it using token index 0.  For the test set, use a batch size of 1 and do not apply a maximum sequence length, so that the model will process the entire review text to classify it.
1. **1D CNN model.**  Create a 1-D CNN model.  You should have three 1D convolutional layers, with a kernel size of 3 and SiLU activation.  After the last convolutional layer, apply "global average pooling," i.e., take an average over the sequence.  The result should be a single 100-dimensional vector for each sequence in the batch.  Then use a linear layer to produce the output logits.  
2.	**GRU model.**  Wrap a GRU (``nn.GRU``) in an `nn.Module`  The GRU returns both the outputs and the hidden states, but you only need the outputs (the first tensor that it returns).   Make sure to use set `batch_first=True` in the GRU.  After the GRU, use a linear layer `nn.Linear` to output the class logits.  The module should only return the class logits for the last element in the sequence.

### Report
Your report should include the following:

- **Code explanation**:
    - Briefly explain your solution and any design choices you made that weren’t specified in the instructions.
    - Clearly describe any external sources that were used (e.g. websites or AI tools) and how they were used.
- **Discussion**: 
    - Discuss whether you expected the GRU or CNN to perform better and why, based on the architecture of each model.
    - Compare the CNN and GRU models in terms of test set accuracy.

### Deliverables
Python notebook and report document.  Be sure to "run all" before submitting.
