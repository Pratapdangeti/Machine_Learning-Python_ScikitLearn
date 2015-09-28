__author__ = 'pratapdangeti'


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score
df = pd.read_csv('sms.csv')

x_train_raw,x_test_raw,y_train,y_test = train_test_split(df['message'],df['label'])
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

scores = cross_val_score(classifier,x_train,y_train,cv=5)
print('Train Accuracy :',np.mean(scores),scores)

precisions = cross_val_score(classifier,x_train,y_train,cv=5,scoring='precision')
print('Train Precisions:',np.mean(precisions),precisions)

recalls = cross_val_score(classifier,x_train,y_train,cv=5,scoring='recall')
print('Train Recalls:',np.mean(recalls),recalls)

f1s=cross_val_score(classifier,x_train,y_train,cv=5,scoring='f1')
print('F1',np.mean(f1s),f1s)

tscores = cross_val_score(classifier,x_test,y_test,cv=5)
print('Train Accuracy :',np.mean(tscores),tscores)

tprecisions = cross_val_score(classifier,x_test,y_test,cv=5,scoring='precision')
print('Train Precisions:',np.mean(tprecisions),tprecisions)

trecalls = cross_val_score(classifier,x_test,y_test,cv=5,scoring='recall')
print('Train Recalls:',np.mean(trecalls),trecalls)

tf1s=cross_val_score(classifier,x_test,y_test,cv=5,scoring='f1')
print('F1',np.mean(tf1s),tf1s)

