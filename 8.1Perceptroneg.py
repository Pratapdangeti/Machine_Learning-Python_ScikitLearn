__author__ = 'pratapdangeti'






from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron

categories = ['rec.sport.hockey','rec.sport.baseball','rec.autos']

newsgroups_train = fetch_20newsgroups(subset='train',categories=categories,remove=('headers','footers','quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories,remove=('headers','footers','quotes'))

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(newsgroups_train.data)
x_test = vectorizer.transform(newsgroups_test.data)

classifier = Perceptron(n_iter=100,eta0=0.1)
classifier.fit_transform(x_train,newsgroups_train.target)
predictions = classifier.predict(x_test)
print(classification_report(newsgroups_test.target,predictions))



