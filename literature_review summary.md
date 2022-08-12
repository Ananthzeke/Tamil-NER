# Summary

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
