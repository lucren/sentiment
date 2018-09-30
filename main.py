from numpy import average
from pandas import read_csv
from sklearn.svm import SVC
from gensim.models import KeyedVectors, TfidfModel
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from gensim.scripts.glove2word2vec import glove2word2vec
# load up the embedding space -> I want to use Glove instead of W2V
# https://www.aclweb.org/anthology/D14-1162
glove_file = datapath("/Users/kristaps/Projs/_glove/glove.42B.300d.txt")
tmp_file = get_tmpfile("test_word2vec.txt")
# gensim doesn't have a class to house Glove vectors so they have to be converted
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
# load up the data files
train_data = read_csv("data/train.csv")
test_data = read_csv("data/test.csv")
test_data["category"] = -1
# I know I read a blog post from the guys that made nltk stating that their
# experiments showed that stoplists result in the loss of precision when it
# comes to sentiment analysis - http://www.lrec-conf.org/proceedings/lrec2014/pdf/292_Paper.pdf
# could yield some extra insight
custom_filters = [
    lambda x: x.lower(),
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
]
# turn pandas into a list of dictionaries
prep_data = lambda x: [{
    "id": entry[0],
    "text": list(filter(lambda x: x in model.vocab, preprocess_string(entry[1], custom_filters))),
    "class": entry[2]
} for entry in x]
test_array = prep_data(test_data[["id", "text", "category"]].values)
train_array = prep_data(train_data[["id", "text", "category"]].values)
# going for tfidf (https://cs.stanford.edu/~quocle/paragraph_vector.pdf) for
# the time being, it should be fine, but it doesn't take the order of words
# into account - should look at http://aclweb.org/anthology/P/P16/P16-1089.pdf
# and https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf which should perform
# better in this case
c_dictionary = Dictionary([doc["text"] for doc in train_array])
c_corpus = [c_dictionary.doc2bow(doc["text"]) for doc in train_array]
tfidfmodel = TfidfModel(c_corpus)
def sentence_to_vec(data):
    """
    Turns sentences into their representative vectors

    data    - a list of dictionaries containing info our sentneces

    return  - the same list of dictionaries now containing the vectors as well
    """
    for i in range(len(data)):
        word_vectors = []
        for j in range(len(data[i]["text"])):
            if len(c_dictionary.doc2bow([data[i]["text"][j]])) > 0 and len(tfidfmodel[[c_dictionary.doc2bow([data[i]["text"][j]])[0]]][0]) > 1:
                # dealing with the words that we've never seen before
                word_vectors.append(
                    model.wv[data[i]["text"][j]]
                    *tfidfmodel[[c_dictionary.doc2bow([data[i]["text"][j]])[0]]][0][1]
                )
        if len(word_vectors) == 0:
            # need to figure out what I'm doing with these ones
            # I should probably drop them
            data[i]["vector"] = 300*[0]
        else:
            data[i]["vector"] = average(word_vectors, axis=0)
    return data
train_array = sentence_to_vec(train_array)
test_array = sentence_to_vec(test_array)
# gathering up the data for the sklearn models
X = [e["vector"] for e in train_array]
Y = [e["class"] for e in train_array]
tX = [e["vector"] for e in test_array]
# choosing support vector machines for the time being
# CNN's are probably a good option http://www.aclweb.org/anthology/D14-1181
clf = SVC(gamma='auto')
clf.fit(X, Y)
nY = clf.predict(X)
tY = clf.predict(tX)
# make a confusion matrix
confusion = {
    0: {0: 0, 1: 0},
    1: {0: 0, 1: 0}
}
for i in range(len(Y)):
    confusion[Y[i]][nY[i]] += 1
print(confusion)
print(clf.score(X, Y))
# write the data to files
train_data["pred"] = nY
test_data["pred"] = tY
test_data.to_csv("test_data.csv", index=False)
test_data[["id", "pred"]].to_csv("resp.csv", index=False)
train_data.to_csv("train_data.csv", index=False)
