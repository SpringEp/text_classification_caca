from dataset import DATASET
import json
from sklearn.datasets import load_files
from nltk.corpus import stopwords

movie_data = load_files(DATASET.root)

X, y = movie_data.data, movie_data.target
from nltk.stem import WordNetLemmatizer
import re
from sklearn.datasets import load_files
from dataset import DATASET


# movie_data = load_files(DATASET.pos)
# X, y = movie_data.data, movie_data.target

stemmer = WordNetLemmatizer()
documents = []

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

sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
freq = words_freq[:50]
print(freq)

# with open(DATASET.root / 'token_matching.json', 'w') as file:
#     keywords = json.dump(keywords_dict, file)
