from copy import deepcopy
import pandas as pd
import numpy as np
from utils.metrics import f1_max_threshold
from sklearn.metrics import recall_score, precision_score
from utils.metrics import get_curve_metrics


def iterative_clf(clf, X, y, clfs_equal_func, n_iter=20):
    """
    input: clf Classifier, X Dataset, y Class labels, clfs_equal_funcfunction to qqual between 2 clfs, n_iter default=20
    calculte precision, recall, f1 and threshold for each point in the precision-recall curve
    return: tuple(precisions, recalls, f1_scores, thresholds)
    Note: clf must implement 'score_samples' method.
    """
    precisions, recalls, f1_scores, thresholds = get_curve_metrics(clf, X, y)
    t, p, r, f1 = f1_max_threshold(precisions, recalls, f1_scores, thresholds)
    recalls = [r]
    precisions = [p]
    for i in range(n_iter):
        scores = clf.score_samples(X)
        tmp_s = pd.Series(scores)
        anomaly_indices=tmp_s[tmp_s<=t].index

        y_pred = scores.copy()
        y_pred[scores>t] = 0
        y_pred[scores<=t] = 1
        recalls.append(recall_score(y, y_pred))
        precisions.append(precision_score(y, y_pred))

        tmp_X = X.loc[~X.index.isin(anomaly_indices)]
        tmp_clf = deepcopy(clf).fit(tmp_X)
        if clfs_equal_func(clf, tmp_clf):
            break
        else:
            clf = tmp_clf
    return tmp_clf, recalls, precisions





def svms_equal(clf1, clf2):
    if np.array_equal(clf1.support_vectors_, clf2.support_vectors_) and \
            np.array_equal(clf1.dual_coef_, clf2.dual_coef_):
        return True
    return False
