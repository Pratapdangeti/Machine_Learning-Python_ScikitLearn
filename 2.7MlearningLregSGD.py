__author__ = 'pratapdangeti'


import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
data = load_boston()
print(data)
x_train,x_test,y_train,y_test = train_test_split(data.data,data.target)

x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_train=x_scaler.fit_transform(x_train)
y_train=y_scaler.fit_transform(y_train)
x_test = x_scaler.transform(x_test)
y_test=y_scaler.transform(y_test)

regressor = SGDRegressor(loss='squared_loss')
scores=cross_val_score(regressor,x_train,y_train,cv=5)
print('Cross Validation r-squared scores:',scores)
print('Average cross validation r-squared score',np.mean(scores))
regressor.fit_transform(x_train,y_train)
print('Test set r-squared score',regressor.score(x_test,y_test))






