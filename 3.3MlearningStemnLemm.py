__author__ = 'pratapdangeti'

# import nltk
# nltk.download()

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering','v'))
print(lemmatizer.lemmatize('gathering','n'))

from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
print(stemmer.stem('gathering'))
