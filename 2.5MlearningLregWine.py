__author__ = 'pratapdangeti'

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split

df = pd.read_csv('winequality-red.csv',sep=';')
x=df[list(df.columns)[:-1]]
y=df['quality']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.6)



regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_predictions = regressor.predict(x_test)
print('R squared:',regressor.score(x_test,y_test))

print(x.describe())
print(x_train.describe())
print(x_test.describe())
