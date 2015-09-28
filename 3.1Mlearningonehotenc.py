__author__ = 'pratapdangeti'


from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [
    {'city':'New york'},
    {'city':'San Francisco'},
    {'city':'Chapel Hill'}
]

print(onehot_encoder.fit_transform(instances).toarray())

