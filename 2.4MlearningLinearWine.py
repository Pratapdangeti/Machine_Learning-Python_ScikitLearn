__author__ = 'pratapdangeti'


import pandas as pd
df = pd.read_csv('winequality-red.csv',sep=';')
print(df.describe())
print(df.corr())

import matplotlib.pylab as plt
plt.scatter(df['alcohol'],df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol against Quality')

plt.show()

