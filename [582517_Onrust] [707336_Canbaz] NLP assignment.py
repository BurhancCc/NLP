import nltk
import pandas as pd
import gensim
import spacy
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sn
import re
import numpy as np
import sys
from nltk.stem import PorterStemmer
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.lm import counter
from collections import Counter
from os import listdir
from os import walk
from os.path import isfile, join
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from itertools import chain, repeat
from scipy.sparse import coo_matrix
from math import factorial
from gensim.models import Word2Vec, KeyedVectors
from keyphrase_vectorizers import KeyphraseCountVectorizer
from rake_nltk import Rake
import string
from itertools import chain
from keybert import KeyBERT

# Root folder where the text files are located
mypath = r'C:\Users\burha\Documents\Projecten\NLP\sentimentLabelledSentences'

# Retrieving all text files in the root directory
paths = [join(dirpath,f) for (dirpath, dirnames, filenames) in walk(mypath) for f in filenames]

# Fills a list with all the text lines from the text files
linesLists = []
for path in paths:
    with open(path, 'r') as f:
        lines = f.readlines()
        linesLists += lines

# Dividing every line of text from its sentiment value
stop_words = stopwords.words('english')
lineAndSentimentList = []
for line in linesLists:
    newLine = line.replace("\n", "").split("\t")
    word_tokens = word_tokenize(newLine[0])
    filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    # Removing punctiation marks
    punctiationMarks = re.compile(r"[\]\[.,!-:?';`]", re.IGNORECASE)
    filtered_sentence = [w for w in filtered_sentence if not set(re.findall(punctiationMarks, w)) == set(w)]
    # Removing digits
    digits = re.compile(r"[0123456789]", re.IGNORECASE)
    filtered_sentence = [w for w in filtered_sentence if not set(re.findall(digits, w)) == set(w)]
    # Replacing contractions with their respective corresponding words
    contractions = {"'s": "is", "'d": "would", "'m": "am", "'ve": "have", "n't": "not", "'re": "are", "'ll": "will"}
    for i in range(len(filtered_sentence)):
        if filtered_sentence[i] in contractions:
            filtered_sentence[i] = contractions[filtered_sentence[i]]
    new_sentence = ""
    for w in filtered_sentence:
        new_sentence = new_sentence + w + " "
    newLine[0] = new_sentence
    if newLine[1] == "1":
        newLine[1] = "Positive"
    elif newLine[1] == "0":
        newLine[1] = "Negative"
    lineAndSentimentList.append(newLine)
print(lineAndSentimentList)
df_lineAndSentiment = pd.DataFrame(data = lineAndSentimentList, columns = ['Lines', 'Sentiment'])


# Creating a list of all the words    
wordsList = [word for words in [line for line in [lines[0].split(" ")[0:-1] for lines in lineAndSentimentList]] for word in words]

# Creating a list of all the unique words)
uniqueWordsList = set(wordsList)

# Corpus analysis
print("Number of words: " + str(len(wordsList)))
print("Number of unique words: " + str(len(set(wordsList))))

# Frequency distributions
# Plot the frequency distribution of 30 words with cumulative = True
document = FreqDist(wordsList)
document.plot(30, cumulative=True, title= "distribution")
print("Top 10 most common words")
for i in range(0, 10):
    print(str(i+1) + ". " + document.most_common(10)[i][0] + " occurs " + str(document.most_common(10)[i][1]) + " times.")

# This part of the code will creat the bag of words vector for each line
cv = CountVectorizer(input= 'content', stop_words = 'english')
bow = cv.fit_transform(df_lineAndSentiment['Lines'])

# creating our train, and test set
X_train, X_test, y_train, y_test = train_test_split(bow.toarray(), df_lineAndSentiment['Sentiment'], test_size=0.2, random_state = 5)

# Function for hyperparameter tuning for Logistic regression
def logRegGridSearch(Xx, Yy, v=1):
    parameters = {'penalty':['l1', 'l2', 'elasticnet', None],
        'C':[ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', 'newton-cholesky'],
        'max_iter': [1000]}

    grid_search = GridSearchCV(estimator = LogisticRegression(),  
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 5,
                            verbose=v,
                            n_jobs=-1)

    grid_result = grid_search.fit(Xx, Yy) 

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # this prints all the different hyperparameter combinations
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    # this prints the best hyperparameter combination
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # This prints the amount of time the hyperparameter tuning process took
    elapsedTime = sum([sum(grid_result.cv_results_['mean_fit_time']) + sum(grid_result.cv_results_['mean_score_time'])])
    amountOfCombinations = len(params)
    print("Total elapsed time: " + str(elapsedTime) + " seconds / ~" + str(int(elapsedTime//60.0)) + " minutes for "
        + str(amountOfCombinations) + " sets of parameters.")

# Hyperparameter tuning for Logistic regression
#logRegGridSearch(X_train, y_train)

# We train our Logistic regression model here
logreg=LogisticRegression(C = 1, penalty = 'l2', solver = 'liblinear')

# fit the model with data
logreg.fit(X_train, y_train)
print("score", logreg.score(X_test, y_test))

y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(cnf_matrix)

# Function for presenting the confusion matrix as a heatmap
def heatmapPresenter(matrix, class_names):
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sn.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# Presenting the confusion matrix as a heatmap
heatmapPresenter(cnf_matrix, set(df_lineAndSentiment['Sentiment']))

# Rearranging the data in to comply with Tf.idf vectorizer
tf_df = pd.DataFrame.from_dict({"Line": [lineAndSentimentList[i][0] for i in range(len(lineAndSentimentList))],
    "Sentiment": [lineAndSentimentList[i][1] for i in range(len(lineAndSentimentList))]})

tf_X = tf_df['Line']
tf_y = tf_df['Sentiment']

# Creating train & test sets for Tf.idf
tf_X_train, tf_X_test, tf_y_train, tf_y_test = train_test_split(tf_X, tf_y, test_size=0.2, random_state = 5)

tf_vectorizer = TfidfVectorizer(lowercase=True)
tf_X_train = tf_vectorizer.fit_transform(tf_X_train)
tf_X_test = tf_vectorizer.transform(tf_X_test)

# Hyperparameter tuning for Logistic regression with tf.idf sets
#logRegGridSearch(tf_X_train, tf_y_train)

# Happens to be the same model as above. Fitting the model with data
logreg.fit(tf_X_train, tf_y_train)
print("score", logreg.score(tf_X_test, tf_y_test))

tf_y_pred = logreg.predict(tf_X_test)

cnf_matrix = metrics.confusion_matrix(tf_y_test, tf_y_pred)

print(cnf_matrix)

# Presenting the confusion matrix as a heatmap
heatmapPresenter(cnf_matrix, set(tf_y))


# Retrieving all sentences again as to have full sentences for keyphrase extraction
sentences = [line[0:-3].lower().translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))
    for line in linesLists]

# Also putting the sentiments back in again to make a dataset after keyphrase extraction
sentencesAndSentiments = [[sentences[i], tf_y[i]] for i in range(len(sentences))]
# Keyphrase extraction function
keybert = KeyBERT()
# Extracting keyphrases
keyWords = []
for s in sentences:
    keyWords.append(keybert.extract_keywords(s, stop_words='english', keyphrase_ngram_range=(1,1)))
for i in range(len(keyWords)):
    for j in range(len(keyWords[i])):
        keyWords[i][j] = keyWords[i][j][0]

keyWords = [" ".join(keys) for keys in keyWords]

extracted_sentencesAndSentiments = [[keyWords[i], tf_y[i]] for i in range(len(keyWords)) if keyWords[i] != '']
# 8 (~0.26% of) sentences were lost during this process.


# Creating a dataframe based off keyphrase extracted sentences & their respective sentiments
extracted_df = {"Lines": [sentence[0] for sentence in extracted_sentencesAndSentiments], "Sentiment": [sentence[1]
    for sentence in extracted_sentencesAndSentiments]}
extracted_df = pd.DataFrame.from_dict(extracted_df)

# Preparing data for logistic regression
cv = CountVectorizer(input= 'content', stop_words = 'english')
key_bow = cv.fit_transform(extracted_df['Lines'])
keyphrases_df = pd.DataFrame.sparse.from_spmatrix(key_bow)

key_X = key_bow.toarray()
key_y = extracted_df["Sentiment"]

key_X_train, key_X_test, key_y_train, key_y_test = train_test_split(key_X, key_y, test_size=0.2, random_state = 5)

# Hyperparameter tuning for logistic regression with keyphrase extracted sentences
#logRegGridSearch(key_X_train, key_y_train, v=3)

# Happens to be the same model as above. Training the model with keyphrase-extracted data
key_logreg = LogisticRegression(C = 1, penalty = 'l2', solver = 'liblinear')
key_logreg.fit(key_X_train, key_y_train)
print("score", key_logreg.score(key_X_test, key_y_test))

# Measuring test set accuracy
key_y_pred = key_logreg.predict(key_X_test)
cnf_matrix = metrics.confusion_matrix(key_y_test, key_y_pred)

print(cnf_matrix)

# Presenting the confusion matrix as a heatmap
heatmapPresenter(cnf_matrix, set(key_y))

# In this part we will test the performance of the stemmed dataset versus the original bow
# First we will stem everyword
ps = PorterStemmer()
df_stemmed = df_lineAndSentiment
for line in df_stemmed['Lines']:
    word_tokens = word_tokenize(line)
    new_sentence = ""
    for word in word_tokens:
        word = ps.stem(word)
        new_sentence = new_sentence + " " + word
    df_stemmed['Lines'] = df_stemmed['Lines'].replace(line, new_sentence)


# This part of the code will creat the bag of words vector for each line
cv = CountVectorizer(input= 'content', stop_words = 'english')
bowStem = cv.fit_transform(df_stemmed['Lines'])

# creating our train, and test set
Xstem_train, Xstem_test, ystem_train, ystem_test = train_test_split(bowStem.toarray(), df_stemmed['Sentiment'], test_size=0.2, random_state = 5)

# Hyperparameter tuning for Logistic regression
#logRegGridSearch(Xstem_train, ystem_train)

# We train our Logistic regression model here
logregStem=LogisticRegression(C = 1, penalty = 'l2', solver = 'liblinear')

# fit the model with data
logregStem.fit(Xstem_train, ystem_train)
print("score", logregStem.score(Xstem_test, ystem_test))

ystem_pred = logregStem.predict(Xstem_test)

cnf_matrix_stem = metrics.confusion_matrix(ystem_test, ystem_pred)

print(cnf_matrix_stem)

# Presenting the confusion matrix as a heatmap
heatmapPresenter(cnf_matrix_stem, set(ystem_pred))

# In this part we will do an experiment with n-grams
# This part of the code will creat the bag of words vector for each line
df_n_grams = df_lineAndSentiment
cvngrams = CountVectorizer(input= 'content', stop_words = 'english', ngram_range = (1, 2))
bowngram = cv.fit_transform(df_n_grams['Lines'])

# creating our train, and test set
Xngram_train, Xngram_test, yngram_train, yngram_test = train_test_split(bowngram.toarray(), df_n_grams['Sentiment'], test_size=0.2, random_state = 5)

# Hyperparameter tuning for Logistic regression
#logRegGridSearch(Xngram_train, yngram_train)

# We train our Logistic regression model here
logregngram=LogisticRegression(C = 1, penalty = 'l2', solver = 'liblinear')

# fit the model with data
logregngram.fit(Xngram_train, yngram_train)
print("score", logregngram.score(Xngram_test, yngram_test))

yngram_pred = logregngram.predict(Xngram_test)

cnf_matrix_ngram = metrics.confusion_matrix(yngram_test, yngram_pred)

print(cnf_matrix_ngram)

# Presenting the confusion matrix as a heatmap
heatmapPresenter(cnf_matrix_ngram, set(yngram_pred))


# Tokenize every sentence
df_embedding = df_lineAndSentiment
token_list = []
for line in df_embedding['Lines']:
    word_tokens = word_tokenize(line)
    token_list.append(word_tokens)
df_embedding['Lines'] = token_list
print(df_embedding['Lines'])

# creating our train, and test set
Xvector_train, Xvector_test, yvector_train, yvector_test = train_test_split(df_embedding['Lines'], df_embedding['Sentiment'], test_size=0.2, random_state = 5)


# Create CBOW model
embeddingsSize = 128
model = Word2Vec(sentences=df_lineAndSentiment['Lines'], vector_size=embeddingsSize, window=5, min_count=1, workers=4)

def getVectors(dataset):
  singleDataItemEmbedding=np.zeros(embeddingsSize)
  vectors=[]
  for dataItem in dataset:
    wordCount=0
    for word in dataItem:
      if word in model.wv:
        singleDataItemEmbedding=singleDataItemEmbedding+model.wv[word]
        wordCount=wordCount+1

    vectors.append(singleDataItemEmbedding)
  return vectors

Xvector_train=getVectors(Xvector_train)
Xvector_test=getVectors(Xvector_test)

# Hyperparameter tuning for Logistic regression
#logRegGridSearch(Xvector_train, yvector_train)

# We train our Logistic regression model here
logregvector=LogisticRegression(C = 1, penalty = 'l2', solver = 'liblinear')

# fit the model with data
logregvector.fit(Xvector_train, yvector_train)
print("score", logregvector.score(Xvector_test, yvector_test))

yvector_pred = logregvector.predict(Xvector_test)

cnf_matrix_vector = metrics.confusion_matrix(yvector_test, yvector_pred)

print(cnf_matrix_vector)

# Presenting the confusion matrix as a heatmap
heatmapPresenter(cnf_matrix_vector, set(yvector_pred))