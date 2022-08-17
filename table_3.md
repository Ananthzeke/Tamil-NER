# Character representation:

| Technique      | Max. F1 Measure |
| -------------- | --------------- |
| LSTM           | 93.37% [134]    |
| CNN            | 93.50% [132]    |
| GRU            | 91.93% [21]     |
| LSTM - LM      | 93.09% [107]    |
| CC - LSTM - LM | 92.22% [103]    |

# Word representation:

| Technique           | Max. F1 Measure |
| ------------------- | --------------- |
| trainrd on PubMed   | 71.01% [94]     |
| Trained on Gigaword | 90.69% [88]     | 
| Random              | 96.86% [95]     | 
| Trained on NYT      | 49.50% [90]     | 
| SENNA               | 91.93% [21]     | 
| Google word2vec     | 86.26% [124]    |
| GloVe               | 93.50% [132]    |
| SSKIP               | 91.73% [109]    |
| Twitter Word2vec    | 41.86% [114]    |
| WordPiece           | 92.80% [119]    |

# Hybrid representation

| Technique                       | Max. F1 Measure |
| ------------------------------- | --------------- |
| POS                             | 89.86% [17]     |
| Word Shape                      | 90.65% [91]     |
| Spelling, n-gram, gazetteer     | 90.10% [18]     |
| Capitalization, lexicons        | 91.62% [20]     |
| FOFE                            | 91.17% [116]    |
| Syntactic                       | 40.42% [113]    |
| POS, topics                     | 41.81% [115]    |
| Images                          | 52.40% [118]    |
| Lexical                         | 91.73% [109]    |
| Segment, Position               | 92.80% [119]    |
| LM                              | 91.93% [21]     |
| POS, gazetteer                  | 90.87% [142]    |
| ELMo, dependancy                | 92.94% [110]    |
| ELMo, gazetteer                 | 92.75% [108]    |
| ELMo, POS                       | 92.28% [133]    |
| BERT                            | 93.33% [138]    |
| BERT, document-level embeddings | 93.37% [134]    |
| BERT, global embedding          | 93.47% [135]    |
| Cloze-style LM embeddings       | 93.50% [132]    |
| Plooled contextual embeddings   | 93.47% [136]    |

# Context encoder

| Technique   | Max. F1 Measure |
| ----------- | --------------- |
| CNN         | 90.87% [142]    |
| GRU         | 93.47% [132]    |
| LSTM        | 96.86% [95]     |
| 1D CNN      | 90.65% [91]     |
| BRNN        | 87.21% [98]     |
| MLP         | 91.17% [116]    |
| Transformer | 92.80% [119]    |
| RNN         | 93.47% [136]    |

# Tag decoder

| Technique          | Max. F1 Measure |
| ------------------ | --------------- |
| CRF                | 93.50% [132]    |
| GRU                | 93.47% [135]    |
| Pointer Network    | 96.86% [95]     |
| LSTM               | 75.00% [96]     |
| Softmax            | 92.80% [119]    |
| Reranker           | 91.62% [102]    |
| Semi-CRF           | 91.38% [143]    |
| Softmax +Dice Loss | 93.33% [138]    |
