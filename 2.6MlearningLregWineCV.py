__author__ = 'pratapdangeti'


import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
df = pd.read_csv('winequality-red.csv',sep=';')
x=df[list(df.columns)[:-1]]
y=df['quality']

regressor = LinearRegression()
scores=cross_val_score(regressor,x,y,cv=5)
print(scores.mean(),scores)


