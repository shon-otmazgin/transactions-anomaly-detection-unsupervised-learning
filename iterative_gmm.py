import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from utils.scale_and_split import split_train_test, RANDOM_STATE
from utils.visualizations import plot_recall_precision_curve_samples_scores, plot_recall_precision_change
from utils.iterative_method import iterative_clf
from utils.metrics import get_curve_metrics, f1_max_threshold, x_test_evalute_report

N_COMPONENTS = 3
N_INIT = 2
DATA_SET_PATH = 'dataset/creditcard.csv'

# loading  the data set
df = pd.read_csv(DATA_SET_PATH)


def gmms_equal(clf1, clf2):
    if np.array_equal(clf1.means_, clf2.means_) and \
            np.array_equal(clf1.covariances_, clf2.covariances_) and \
            np.array_equal(clf1.weights_, clf2.weights_):
        return True
    return False


if __name__ == '__main__':
    X, X_test, y, y_test = split_train_test(df=df)

    print('Starting initialization step...')
    clf = GaussianMixture(n_components=N_COMPONENTS, covariance_type='full', random_state=RANDOM_STATE, n_init=N_INIT)
    print(clf)
    clf.fit(X)
    print('Plotting Precision-Recall and scores plots for initialization step...')
    plot_recall_precision_curve_samples_scores(clf=clf, X=X, y=y, clf_name='GMM')

    print()
    print('Starting iterative method... it takes few minutes.')
    clf, recalls, precisions = iterative_clf(clf=clf, X=X, y=y, clfs_equal_func=gmms_equal, n_iter=20)
    print('Plotting Precision-Recall change through iteration')
    plot_recall_precision_change(recalls=recalls, precisions=precisions)

    print('Plotting Precision-Recall and scores plots after iterative method...')
    precisions, recalls, f1_scores, thresholds = get_curve_metrics(clf, X, y)
    t, p, r, f1 = f1_max_threshold(precisions, recalls, f1_scores, thresholds)
    plot_recall_precision_curve_samples_scores(clf=clf, X=X, y=y, clf_name='GMM')
    print()
    print('Test set classification report:')
    x_test_evalute_report(clf=clf, X_test=X_test, y_test=y_test, threshold=t)