__author__ = 'pratapdangeti'




import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score

df = pd.read_csv('SMSSpamCollection',delimiter = '\t',header = None)
# print(df.head)
print('Number of spam messages :',df[df[0]=='spam'][0].count())
print('Number of ham messages:',df[df[0]=='ham'][0].count())

x_train_raw,x_test_raw,y_train,y_test = train_test_split(df[1],df[0])

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)

classifier = LogisticRegression()
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)


# for i, prediction in enumerate(predictions[:5]):
#     print(prediction,x_test_raw[:i])

from sklearn.metrics import accuracy_score
print('Accuracy scores:',accuracy_score(y_test,predictions))


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

conf_matrix=confusion_matrix(y_test,predictions)
print(conf_matrix)
plt.matshow(conf_matrix)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()






