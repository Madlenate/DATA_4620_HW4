# DATA 4620 HW4: Sentiment Analysis

**Author:** Nathan Madlansacay
**Date:** 11/8/2026

Binary sentiment classification of 50,000 IMDB movie reviews (positive vs. negative), comparing a **1D CNN** and a **GRU**.

## Results at a Glance

| Model  | Test Accuracy | Final Train Accuracy | Final Train Loss | Training Time (CPU) |
|--------|---------------|----------------------|------------------|---------------------|
| 1D CNN | **90.78%**    | 86.37%               | 0.3113           | ~24 min             |
| GRU    | 90.02%        | 86.93%               | 0.3040           | ~35 min             |

The 1D CNN finished slightly ahead of the GRU and trained faster.

## Overview

Both models read BERT BPE tokens through an embedding layer and use the same shared hyperparameters. The only difference between them is the architecture after the embedding.

The notebook tokenizes every review once, splits the data, wraps it in a custom `Dataset`, and trains both models with one shared training function.

## Preprocessing

- **Tokenization:** Every review is tokenized once with the BERT (cased) BPE tokenizer (vocabulary size 28,996).
- **Split:** `train_test_split` creates a 90/10 split, stratified on the labels so both sets stay 50/50 positive and negative.
- **`TokenDataset`:** A custom PyTorch `Dataset` that serves the tokenized reviews.
  - **Training:** A review longer than 100 tokens is cut to a random 100-token window, chosen fresh each time the item is loaded. A shorter review is padded to 100 tokens.
  - **Test:** The full review is returned with no cropping or padding.
- **DataLoaders:** Training uses batches of 64 with shuffling. Testing uses a batch size of 1, so every review keeps its natural length.

## Notable Design Choices

- **Batch size 64:** The default setting, a middle ground between speed and gradient noise.
- **No special tokens:** BERT's special tokens (`[CLS]`, `[SEP]`) are meant for BERT, not these models, so they are left out. Random crops are used to cut long reviews down instead.
- **Padding embedding:** `padding_idx` keeps the pad vector fixed at zero, so padding carries no learned meaning.
- **Output and loss:** 2 logits with `CrossEntropyLoss`.

## Model Architectures

### 1D CNN

1. `nn.Embedding(28996, 100)` maps each token ID to a 100-dimensional vector.
2. The embeddings are transposed from `(batch, length, channels)` to `(batch, channels, length)`, because `Conv1d` expects channels second.
3. Three `Conv1d(100, 100, kernel_size=3, padding=1)` layers, each followed by `SiLU`. The padding keeps the sequence length unchanged.
4. Global average pooling (mean over the length dimension) turns the sequence into one 100-dimensional vector. This also lets the same model handle 100-token training batches and full-length test reviews.
5. `nn.Linear(100, 2)` outputs the two class logits.

### GRU

1. The same embedding layer.
2. `nn.GRU(100, 100, num_layers=3, batch_first=True)`. Only the first returned tensor (the outputs at every time step) is kept; the hidden states are discarded.
3. The output at the last time step, `out[:, -1, :]`, summarizes the whole sequence and goes through `nn.Linear(100, 2)` to produce the logits.

## Conclusion

At first I expected the GRU to perform better because of its structure. The GRU's ability to carry memory through a review, deciding what to keep and what to forget, should help it tell a negative review from a positive one and learn faster and more efficiently.

In the end, both models reached about 90% test accuracy. The 1D CNN scored slightly higher (90.78% vs. 90.02%) and trained faster. For IMDB sentiment, local phrase patterns pooled across the whole review work as well as sequential modeling, at a lower cost.

## Possible Improvements

- Seed Python's `random` module for reproducible crops.
- Run several seeds to check whether the accuracy gap is real.
- Pad on the left for the GRU, so the last time step is a real token rather than padding.
- Train on longer sequences.

## External Source Usage

Claude was used to polish the code, to explain parts of the code and teach new techniques, and to help structure the code more efficiently.
