from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from time import time
from utils import MNISTDataSingleton


def run():
    t0 = time()
    print('loading data')
    X_train_pca = MNISTDataSingleton().X_train_pca
    y_train = MNISTDataSingleton().y_train

    X_test_pca = MNISTDataSingleton().X_test_pca
    y_test = MNISTDataSingleton().y_test

    print('data loaded in time {}'.format(time() - t0))

    param_grid = {
        'C': [1e3, 5e3, 5e4, 1e5],
        'gamma': [0.0001, 0.005, 0.01, 0.1],
        'kernel': ['rbf', 'linear'],
    }

    clf = GridSearchCV(SVC(), param_grid=param_grid, n_jobs=6, verbose=1)

    print('training classifier')
    t0 = time()
    clf.fit(X_train_pca, y_train)
    print('classifier trained in time {}'.format(time() - t0))
    print('best estimator: \n {}'.format(clf.best_estimator_))

    y_pred = clf.predict(X_test_pca)

    score = f1_score(y_pred=y_pred, y_true=y_test, labels=list(range(10)), average=None)

    print('score is {}'.format(score))

    print('saving model')
    joblib.dump(clf, 'svm_clf.pkl')
    print('model saved')


if __name__ == '__main__':
    run()