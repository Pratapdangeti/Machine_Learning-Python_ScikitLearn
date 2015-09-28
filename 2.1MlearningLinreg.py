__author__ = 'pratapdangeti'


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


x=[[6],[8],[10],[14],[18]]
y=[[7],[9],[13],[17.5],[18]]

model=LinearRegression()
model.fit(x,y)

print 'A 12" pizza should cost: $%.2f'%model.predict([12])[0]
print model.intercept_ ,model.coef_
print "Training data set R-squared:",model.score(x,y)

x_test=[[8],[9],[11],[16],[12]]
y_test=[[11],[8.5],[15],[18],[11]]

print 'Residual sum of squares: %.2f'%np.mean((model.predict(x)-y)**2)

print "R squared:%.4f"%model.score(x_test,y_test)

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in Inches')
plt.ylabel('Price in Dollars')
plt.plot(x,y,'k.')
plt.axis([0,25,0,25])
plt.grid(True)
plt.plot(x,model.predict(x),lw=2.0)
plt.show()

