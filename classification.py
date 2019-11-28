import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset import DATASET

movie_data = load_files(DATASET.root)
X, y = movie_data.data, movie_data.target

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents)

# sum_words = X.sum(axis=0)
# words_freq = [(word, sum_words[0, idx]) for word, idx in ]
# words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
# freq = words_freq[:30]
# print(freq)



keywords = ['scene',
                'good',
                'make',
                'story',
                'would']

feature_names = vectorizer.get_feature_names()

# weights = np.ones(len(feature_names))
# for key, value in word_weights.items():
#     weights[feature_names.index(key)] = value

weight_factor = 5

for keyword in keywords:
    position = vectorizer.vocabulary_[keyword]
    print(position)
    print(X[0,position])
    X[:,position] *= weight_factor
    print(X[0,position])
    print('---------------------------')

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X.toarray()).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))