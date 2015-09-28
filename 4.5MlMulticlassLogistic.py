__author__ = 'pratapdangeti'

# Data exploring and understanding step
# import pandas as pd
# df = pd.read_csv('train.tsv',header = 0,delimiter='\t')
# print(df.count())
# print(df.head())

# print(df['Phrase'].head(10))
# print(df['Sentiment'].describe())
# print(df['Sentiment'].value_counts())
# print(df['Sentiment'].value_counts()/df['Sentiment'].count())


import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
# import matplotlib.pyplot as plt


def main():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    parameters = {
        'vect__max_df': (0.25, 0.5, 0.75),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__use_idf': (True, False),
        'clf__C': (0.1, 1, 10),

        # 'vect__stop_words': ('english', None),
        # 'vect__max_features': (5000, 10000, None),
        # 'vect__norm': ('l1', 'l2'),
        # 'clf__penalty': ('l1', 'l2'),

    }
    df = pd.read_csv('train.tsv', header=0, delimiter='\t')
    X, y = df['Phrase'], df['Sentiment'].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    # grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print ('Best score: %0.3f' % grid_search.best_score_)
    print ('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print ('\t%s: %r' % (param_name, best_parameters[param_name]))

    predictions = grid_search.predict(X_test)
    print ('Accuracy:', accuracy_score(y_test, predictions))
    print ('Classification Report:', classification_report(y_test, predictions))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    print (cm)

    # plt.matshow(cm)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()
    predictions = np.ones(len(predictions)) * 2
    print ('Accuracy:', accuracy_score(y_test, predictions))
    print ('Degenerate Classification Report:', classification_report(y_test, predictions))

if __name__ == '__main__':
    main()





















