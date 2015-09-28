__author__ = 'pratapdangeti'





import os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import skimage
# import Image


def resize_and_crop(image,size):
    img_ratio = image.size[0]/float(image.size[1])
    ratio = size[0]/float(size[1])
    if ratio>img_ratio:
        image=image.resize((size[0],size[0]*image.size[1]/image.size[0]),)



"""
import os
import numpy as np
import mahotas as mh
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
    X = []
    y = []
    for path, subdirs, files in os.walk('data/English/Img/GoodImg/Bmp/'):
        for filename in files:
            f = os.path.join(path, filename)
            target = filename[3:filename.index('-')]
            img = mh.imread(f, as_grey=True)
            if img.shape[0] <= 30 or img.shape[1] <= 30:
                continue
            img_resized = mh.imresize(img, (30, 30))
            if img_resized.shape != (30, 30):
                img_resized = mh.imresize(img_resized, (30, 30))
            X.append(img_resized.reshape((900, 1)))
            y.append(target)
    X = np.array(X)
    X = X.reshape(X.shape[:2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
    pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
    ])
    parameters = {
        'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
        'clf__C': (0.1, 0.3, 1, 3, 10, 30),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print classification_report(y_test, predictions)


"""
