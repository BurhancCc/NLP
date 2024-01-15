import nltk
import pandas as pd
import gensim
import spacy
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sn
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
import re
import numpy as np
import sys
from scipy.sparse import coo_matrix
from math import factorial

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
    # Removing words that are purely numbers
    numbers = re.compile("[0-9]", re.IGNORECASE)
    filtered_sentence = [w for w in filtered_sentence if not set(re.findall(numbers, w)) == set(w)]
    # Replacing contractions with their respective corresponding words
    contractions = {"'s": "is", "'d": "would", "'m": "am", "'ve": "have", "n't": "not", "'re": "are", "'ll": "will"}
    for i in range(len(filtered_sentence)):
        if filtered_sentence[i] in contractions:
            filtered_sentence[i] = contractions[filtered_sentence[i]]
    newLine[0] = filtered_sentence
    lineAndSentimentList.append(newLine)
print(len(lineAndSentimentList))

# Creating a list of all the words    
wordsList = [word for words in [line for line in [lines[0] for lines in lineAndSentimentList]] for word in words]

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
bow = cv.fit_transform(linesLists)
df = pd.DataFrame.sparse.from_spmatrix(bow)

# This part replaces our sentiment values from 1 and 0 to positive and negative
# For 1 we use positive and for 0 we use negative
sentimentList = []
for sentiment in lineAndSentimentList:
    if sentiment[1] == "1":
        sentimentList.append("Positive")
    elif sentiment[1] == "0":
        sentimentList.append("Negative")

df["Sentiment"] = sentimentList

# Here do we split our attributes between target and not target
X = df.loc[:, df.columns != 'Sentiment']
y = df["Sentiment"]

# creating our train, and test set
X_train, X_test, y_train, y_test = train_test_split(bow.toarray(), sentimentList, test_size=0.2, random_state = 5)

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
heatmapPresenter(cnf_matrix, set(y))

# Rearranging the data in to comply with Tf.idf vectorizer
tf_df = pd.DataFrame.from_dict({"Line": [" ".join(lineAndSentimentList[i][0]) for i in range(len(lineAndSentimentList))],
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


# Function for sorting tf_idf in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    # Use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

tf_doc = tf_X
tf_doc_transformed = tf_vectorizer.transform(tf_X)
sorted_items=sort_coo(tf_doc_transformed.tocoo())

feature_names = cv.get_feature_names_out()
keywords=extract_topn_from_vector(feature_names,sorted_items,99999999999)
# now print the results
print("\nAbstract:")
print(tf_doc)
print("\nKeywords:")
for k in keywords:
    print(k,keywords[k])