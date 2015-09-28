__author__ = 'pratapdangeti'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]

x_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

regressor = LinearRegression()
regressor.fit(x_train,y_train)
xx = np.linspace(0,26,100)
yy=regressor.predict(xx.reshape(xx.shape[0],1))
plt.plot(xx,yy)


quadratic_featurizer = PolynomialFeatures(degree=2)
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic,y_train)

xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))


print(x_train)
print(x_train_quadratic)
print(x_test)
print(x_test_quadratic)


print('Simple linear regression r-squared',regressor.score(x_test,y_test))
print('Quadratic regression r-squared',regressor_quadratic.score(x_test_quadratic,y_test))




plt.plot(xx,regressor_quadratic.predict(xx_quadratic),c='r',linestyle ='--')
plt.title('Pizza price regressed on Diameter')
plt.xlabel('Diameter in Inches')
plt.ylabel('Price in Dollars')
plt.axis([0,25,0,25])
plt.scatter(x_train,y_train)
plt.show()





