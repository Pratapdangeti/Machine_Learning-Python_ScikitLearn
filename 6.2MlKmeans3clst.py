__author__ = 'pratapdangeti'

import numpy as np
from sklearn.cluster import KMeans

from sklearn import metrics
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
plt.subplot(3,2,1)
x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])

X = np.array(zip(x1,x2)).reshape(len(x1),2)
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instances')
plt.scatter(x1,x2)
colors = ['b','g','r','c','m','y','k','b']
markers = ['o','s','D','v','^','p','*','+']
tests = [2,3,4,5,8]
subplot_counter = 1

for t in tests:
    subplot_counter +=1
    plt.subplot(3,2,subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    for i,l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i],x2[i],color=colors[l],marker = markers[l],ls='None')
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.title('K= %s, silhoutte coefficient = %.03f'%(t,metrics.silhouette_score(X,kmeans_model.labels_,metric = 'euclidean')))

plt.show()
