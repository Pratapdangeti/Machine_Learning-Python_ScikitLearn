__author__ = 'pratapdangeti'



from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
    data = fetch_mldata('MNIST original',data_home='data/mnist')
    x,y =data.data,data.target
    x=x/255.0*2-1
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    pipeline = Pipeline([
        ('clf',SVC(kernel='rbf',gamma=0.01,C=100))
    ])
    print(x_train.shape)
    parameters = {
        'clf__gamma':(0.01,0.03,0.1,0.3,1),
        'clf__C':(0.1,0.3,1,3,10,30),
    }
    grid_search = GridSearchCV(pipeline,parameters,n_jobs=2,verbose=1,scoring='accuracy')
    grid_search.fit(x_train[:10000],y_train[:10000])
    print('Best score: %0.3f'% grid_search.best_score_)
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r'%(param_name,best_parameters[param_name]))
    predictions = grid_search.predict(x_test)
    print(classification_report(y_test,predictions))


