import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM

from utils.scale_and_split import split_train_test, RANDOM_STATE
from utils.visualizations import plot_recall_precision_curve_samples_scores, plot_recall_precision_change
from utils.iterative_method import iterative_clf
from utils.metrics import get_curve_metrics, f1_max_threshold, x_test_evalute_report

KERNEL = 'rbf'
NU = 0.01
GAMMA = 'scale'
DATA_SET_PATH = 'dataset/creditcard.csv'

# loading  the data set
df = pd.read_csv(DATA_SET_PATH)


def svms_equal(clf1, clf2):
    '''
    :param clf1: onclasssvm classifier
    :param clf2: onclasssvm classifier
    :return: True if clf1 equal with his estimators to clf2
    '''
    if np.array_equal(clf1.support_vectors_, clf2.support_vectors_) and \
       np.array_equal(clf1.dual_coef_, clf2.dual_coef_):
        return True
    return False


if __name__ == '__main__':
    X, X_test, y, y_test = split_train_test(df=df)

    print('Starting initialization step...')
    clf = OneClassSVM(kernel=KERNEL, nu=NU, gamma=GAMMA)
    print(clf)
    clf.fit(X)
    print('Plotting Precision-Recall and scores plots for initialization step...')
    plot_recall_precision_curve_samples_scores(clf=clf, X=X, y=y, clf_name='OneClassSVM')

    print()
    print('Starting iterative method... it takes few minutes.')
    clf, recalls, precisions = iterative_clf(clf=clf, X=X, y=y, clfs_equal_func=svms_equal, n_iter=20)
    print('Plotting Precision-Recall change through iteration')
    plot_recall_precision_change(recalls=recalls, precisions=precisions)

    print('Plotting Precision-Recall and scores plots after iterative method...')
    precisions, recalls, f1_scores, thresholds = get_curve_metrics(clf, X, y)
    t, p, r, f1 = f1_max_threshold(precisions, recalls, f1_scores, thresholds)
    plot_recall_precision_curve_samples_scores(clf=clf, X=X, y=y, clf_name='OneClassSVM')
    print()
    print('Test set classification report:')
    x_test_evalute_report(clf=clf, X_test=X_test, y_test=y_test, threshold=t)