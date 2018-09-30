# Sentiment

Contains an implementation of Mikolovs average(TF-IDF*vector) algorithm (https://cs.stanford.edu/~quocle/paragraph_vector.pdf) that uses the Glove
embedding space instead of W2V. Sentences that have been represented in this
fashion can be freely fed into classification models - a SVM in this case.

# Further reading / Alternative approaches

* On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter http://www.lrec-conf.org/proceedings/lrec2014/pdf/292_Paper.pdf
* Siamese CBOW: Optimizing Word Embeddingsfor Sentence Representations http://aclweb.org/anthology/P/P16/P16-1089.pdf
* Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf
* Convolutional Neural Networks for Sentence Classification http://www.aclweb.org/anthology/D14-1181
