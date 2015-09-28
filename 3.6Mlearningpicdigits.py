__author__ = 'pratapdangeti'

from sklearn import datasets
digits=datasets.load_digits()
print('Digit:',digits.target[0])
print(digits.images[0])
print('Feature vector: \n',digits.images[0].reshape(-1,64))

