# Summary

Comparision of Previous works on Tamil NER

| Paper                               | Dataset                      | Tagset                     | Features                             | Model       | Result (F1 score) |
| ----------------------------------- | ---------------------------- | -------------------------- | ------------------------------------ | ----------- | ----------------- |
| [10] Pattabhi et al.,   2020        | Twitter Corpus               | AU-KBC Hierarchial tagset  | stat. suffixes using context windows | CRF         | 70.93             |
| [1] Srinivasan &  Subalalitha, 2019 | Groningen Meaning Bank (GMB) | PER, LOC, ORG, Date, Time  | REGEX, MORPH, Context windows        | Naive Bayes | 86.94             |
| [1] Srinivasan &  Subalalitha, 2019 | FIRE 2018 IECSIL             | IECSIL (9)                 | REGEX, MORPH, Context windows        | Naive Bayes | 83.54             |
| [2] Hariharan et al., 2019          | FIRE 2018 IECSIL             | IECSIL (9)                 | FastText Embeddings                  | LSTM        | 94.54             |
| [2] Hariharan et al., 2019          | FIRE 2018 IECSIL             | IECSIL (9)                 | GloVE Embeddings                     | LSTM        | 93.06             |
| [2] Hariharan et al., 2019          | FIRE 2018 IECSIL             | IECSIL (9)                 | Rand Embeddings                      | LSTM        | 87.23             |
| [2] Hariharan et al., 2019          | wiki crawl                   | Wiki (21)                  | FastText Embeddings                  | LSTM        | 91.29             |
| [2] Hariharan et al., 2019          | wiki crawl                   | Wiki (21)                  | GloVE Embeddings                     | LSTM        | 90.06             |
| [2] Hariharan et al., 2019          | wiki crawl                   | Wiki (21)                  | Rand Embeddings                      | LSTM        | 84.03             |
| [8] Theivendiram et al., 2018       | Tamil BBC newspaper articles | PER, LOC, ORG, COUNT, TIME | gazetteer, POS, Orth features        | MIRA        | 81.38             |
| [8] Theivendiram et al., 2018       | Tamil BBC newspaper articles | PER, LOC, ORG, COUNT, TIME | gazetteer, POS, Orth features        | CRF         | 79.13             |
| [9] Abinaya et al., 2015            | NER Track (FIRE)             | PER, LOC, ORG              | Word                                 | RKS         | 86.61 (accuracy)  |
| [9] Abinaya et al., 2015            | NER Track (FIRE)             | PER, LOC, ORG              | Word                                 | SVM         | 81.62 (accuracy)  |
| [9] Abinaya et al., 2015            | NER Track (FIRE)             | PER, LOC, ORG              | Word                                 | CRF         | 87.21 (accuracy)  |
| [4] Jeyashenbagavalli et al., 2014  | private corpus               | NE (7)                     | POS and word level features          | RULE+HMM    | 89.7              |
| [5] Malarkodi et al., 2012          | Online Tourism corpus        | AU-KBC Hierarchial tagset  | Word, POS, chunks                    | CRF         | 70.68             |

NE (7) - PER, ORG, LOC, TIME, MEASURE, NUMBER

# Problem for tamil NLP and suggested solutions

1.  [5] - Agglutination -  Morpholological analysis
3.  [5] - Ambiguity - context features can be used
4.  [5] - Nested Entities - Only the outer entitiy is tagged 
5.  [5] - Name variations - character level embeddings can be used
6.  [5] - Spell variations - character level embeddings can be used
7.  [5] - Capitalization - Rule based systems cannot be used
8.  [5] - Lack of Standard corpora - Use AI4Bharath dataset for comparision
9.  [5] - Morphology - taking into account the morphological data using subword level entities
10. [6] - Partial free word order 

# DL Based approaches Taxonomy
## Distributed Input Representations
- Character Level Representations
  - CNN Based 
  - RNN Based 
    - LSTM
    - GRU
- Word Level Representations - Pretrained representations using unsupervised methods
  - CBOW
  - Continuous Skipgram
  - Word2Vec - Google
  - GloVe - Stanford
  - fastText - Facebook
- Hybrid Representations - Uses additional information
  - gazetteers
  - lexical similarity
  - linguistic dependencies
  - visual features
  - Spelling features
  - context features
## Context Encoders
- CNN - Colobert et al. [17], CNN on BioNER - Yao et al.[94], Wu et al. [12]
- RNN (BLSTM-RE) - Zhou et al. [96] , BLSTM and CNN with sigmoid classifier, both are passed to another LSTM to predict entities
- ID-CNN - larger context using Dialated CNN Block comparable results with Bi-LSTM-CRF - strubell et al. (2017)
- Bidirectional RNN considered as de facto standard for deep context dependent representations of text [91], [97]
- BiLSTM-CRF - Haung et al.
- Deep GRU on both character level and word levels to encode morphology and context information. - Yang et al [106].
- Multiple independent BiLSTM across same input by employing inter-model regularization term, reducing total number of parameters - Gregoric et al. [121]

## Tag Decoders
- MLP + Softmax - predicts tags for words independently based on context dependent representations
- CRF - used in combination with BiLSTM and CNN layers, provides state of art performance
- RNN - Outperforms CRF and faster to train when number of entity types is large
- Pointer Networks - two neural nets one segments and the other labels - Zhai et al.


# Challenges in Deep learning based NER aproaches
- Data annotation 
  - Time Consuming
  - Inconsistencies
  - Nested entities
- Informal text
  - Models trained on formal texts provide better accuracy in formal text
  - but show less accuracy in informal texts such as tweets, comments, etc.
- Unseen entities

# Future Directions in DL NER approaches
- NER with Domain Specific Entities
- Named Entitiy Boundary Detection - Dedicated task for boundary detection of named entities
- Entity linking - Linking an entity with its unique identity in a knowledge base
- NER in informal text with auxilary resources 
  - Usage of gezzeters as additional features does not show an evidence of performance increase
  - Method for incorporating auxilary resources with DL based approaches.
- Making scalable NER models increasing performance and reducing model complexity - model compression and pruning techniques may be used
- Transfer learning with NER
  - Adapting to different domains
  - zero-shot, one-shot and few-shot learning
  - domain mismatch, label mismatch in cross domain techniques.
- Development of Easy to use toolkit for DL based NER

## Models 

- Naive Bayes Classification:
  - Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems. 
  - It is mainly used in text classification that includes a high-dimensional training dataset.
  - It assumes that the occurrence of a certain feature is independent of the occurrence of other features.
  - Bayes's Theorem:
    - Bayes' theorem is used to determine the probability of a hypothesis with prior knowledge. It depends on the conditional probability.
    - Formula: P(A|B) = (p(B|A)*p(A))/P(B)
    - P(A|B) is Posterior probability, P(B|A) is Likelihood probability, P(A) is Prior Probability, P(B) is Marginal Probability.
  - Advantages:
      - Naïve Bayes is one of the fast and easy ML algorithms to predict a class of datasets.
      - It performs well in Multi-class predictions as compared to the other Algorithms.
  - Disadvantages:
      - Naive Bayes assumes that all features are independent or unrelated, so it cannot learn the relationship between features.
  - Maximum accuracy: F1 score – 86.94% (1)
  - Average accuracy: F1 score – 86.94

- Recurrent Neural Network:
  - A recurrent neural network (RNN) is a special type of an artificial neural network adapted to work for time series data or data that involves sequences.
  - RNNs have the concept of ‘memory’ that helps them store the states or information of previous inputs to generate the next output of the sequence.
  - Advantages:
    - Ability to handle sequence data.
    - Ability to handle inputs of varying lengths.
    - Ability to store or ‘memorize’ historical information.
  - Disadvantages:
    - The computation can be very slow.
    - The network does not take into account future inputs to make decisions.
    - Vanishing gradient problem.
  - Long Short Term Memory (LSTM):
    - LSTMs were also designed to address the vanishing gradient problem in RNNs. 
    - LSTM use three gates called input, output and forget gate. Similar to GRU, these gates determine which information to retain.
  - Maximum accuracy: F1 score – 94.54% (2 LSTM)
  - Average accuracy: F1 score – 92.91
  
- Hidden Markov Model:
  - A Hidden Markov Model (HMM) is a class of probabilistic graphical models that allow us to predict a sequence of unknown variables from a set of observed variables.
  - The main goal of HMM is to learn about a Markov chain by observing its hidden states.
  - Advantages:
    - Efficient learning algorithms-learning can take place directly from raw sequence data.
    - Can handle inputs of variable length.
  - Disadvantages:
	  - They cannot express dependencies between hidden states.
  - Maximum accuracy: F1 score – 89.7% (4)
  - Average accuracy: F1 score – 86.35

- Conditional Random Fields:
  - Conditional Random Fields is a class of discriminative models best suited to prediction tasks where contextual information or state of the neighbours affect the current prediction.
  - Advantages:
    - Relaxation of the independence assumption.
  - Maximum accuracy: F1 score – 87.21% (9)
  - Average accuracy: F1 score – 77.2

- Margin-Infused Relaxed Algorithm:
  - MIRA is an online algorithm, which is based on error minimization. It makes use of a matrix to build a model. In each iteration, different matrices are considered by making a small change to the parameters of the earlier matrix and the matrix that makes the lowest error is selected as the final matrix. Likewise, iterations are continued throughout the training data and final matrix is discovered.
  - Advantages:
    - Performs better than CRF when tested using south and South East Asian Languages.
  - Maximum accuracy: F1 score – 81.38% (8)
  - Average accuracy: F1 score – 81.38
	
- Random Kitchen Sink Algorithm:
  - Random Kitchen Sink algorithm is a machine learning  algorithm for classification of nonlinearly separated data set.
  - Random Kitchen Sink uses Radial Basis Function (RBK) kernel which is a real Gaussian function.
  - Advantages:
    - The conventional nonlinear kernel methods use large non-linear data set for training the system. It requires large proportion of data points to be stored for classifying new data point. So, space and time requirement is more for classification. Random Kitchen Sink is an alternative for these conventional nonlinear kernel methods. RKS uses only the feature size and does not consider the number of data points for classification.
  - Maximum accuracy: F1 score – 86.61% (9)
  - Average accuracy: F1 score – 86.61
	
- Support Vector Machine:
	- Support Vector Machine (SVM) is a supervised learning algorithm that generates a hyperplane which separate two classes.
  - Advantages:
    - It is more productive in high dimensional spaces.
  - Disadvantages:
    - Support vector machine algorithm is not acceptable for large data sets.
    - In cases where the number of properties for each data point outstrips the number of training data specimens, the support vector machine will underperform.
  - Maximum accuracy: F1 score – 81.62% (9)
  - Average accuracy: F1 score – 81.62
