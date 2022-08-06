1 - Automated Named Entity Recognition from Tamil Documents

Challenges in tamil:
- partially free word order
- Sentence without SV or O
- No capitalization
- polysemic : single word with multiple meanings
- Ambiguity 
- Lack of resources
  
Features:
 - regex features : for dates
 - morphological features : case markers
 - Contextual features 

Method:
- naive bayes using the above features

Data:
- FIRE corpus

Evalutation:
- F-measure of 83.54%

2 - Named Entity Recognition in Tamil
Language Using Recurrent Based
Sequence Model

Data:
- Wikipedia crawling
- FIRE-2018 IECSIL shared task corpus.
![](./Entitiy%20tag%20statistics.png)

Method:
- LSTM
- No Word2Vec or GloVE - does not consider internal structure of the words
- Uses FastText - considers ngrams thus morphology is learned to some extent
- Model : embedding, LSTM, Dense, Output Softmax

Evalutation:
- ![](NER%20Results.png)

