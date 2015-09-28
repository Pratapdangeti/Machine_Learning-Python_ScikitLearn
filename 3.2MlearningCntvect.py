__author__ = 'pratapdangeti'


from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]

# vectorizer = CountVectorizer()
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

from sklearn.metrics.pairwise import euclidean_distances
counts = vectorizer.fit_transform(corpus).todense()
print('Distance between 1st and 2nd documents:',euclidean_distances(counts[0],counts[1]))
print('Distance between 1st and 3rd documents:',euclidean_distances(counts[0],counts[2]))
print('Distance between 2nd and 3rd documents:',euclidean_distances(counts[1],counts[2]))


