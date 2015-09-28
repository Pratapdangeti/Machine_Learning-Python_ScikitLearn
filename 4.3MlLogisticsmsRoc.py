__author__ = 'pratapdangeti'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import roc_curve,auc
df=pd.read_csv('sms.csv')
x_train_raw,x_test_raw,y_train,y_test = train_test_split(df['message'],df['label'])
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
predictions = classifier.predict_proba(x_test)
false_positive_rate,recall,thresholds = roc_curve(y_test,predictions[:,1])
roc_auc=auc(false_positive_rate,recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,recall,'b',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()












