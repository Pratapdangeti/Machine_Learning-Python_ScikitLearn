__author__ = 'pratapdangeti'



import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib.cm as cm

digits = fetch_mldata('MNIST original',data_home='data/mnist').data

counter = 1

for i in range(1,4):
    for j in range(1,6):
        plt.subplot(3,5,counter)
        plt.imshow((digits[(i-1)*8000+j].reshape((28,28))),cmap = cm.Greys_r)
        plt.axis('off')
        counter+=1
plt.show()

