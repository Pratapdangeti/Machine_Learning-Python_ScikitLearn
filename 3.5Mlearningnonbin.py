__author__ = 'pratapdangeti'

# from sklearn.feature_extraction.text import CountVectorizer
# corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
# vectorizer = CountVectorizer(stop_words='english')
# print(vectorizer.fit_transform(corpus).todense())
# print(vectorizer.vocabulary_)

#TD-IDF

# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = [
#     'The dog ate a sandwich and I ate a sandwich',
#     'The wizard trasnfigured a sandwich'
# ]
#
# vectorizer = TfidfVectorizer(stop_words='english')
# print(vectorizer.fit_transform(corpus).todense())

#Using hashing trick

from sklearn.feature_extraction.text import HashingVectorizer
corpus=['the','ate','bacon','cat']
vectorizer=HashingVectorizer(n_features=6)
print(vectorizer.transform(corpus).todense())

