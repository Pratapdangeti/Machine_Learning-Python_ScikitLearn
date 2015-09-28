__author__ = 'pratapdangeti'





from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM

y=[0,1,1,0]*1000
x=[[0,0],[0,1],[1,0],[1,1]]*1000
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=3)

clf =



