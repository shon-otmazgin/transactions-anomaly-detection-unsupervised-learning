import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from warnings import filterwarnings
filterwarnings('ignore')


def get_gmms_by_clusters(X, n_clusters, iterations, n_init=1):
    '''
    :param X: Data frame of the data
    :param n_clusters: Range of clusters to fit
    :param iterations: Num of fits iteration in each cluster
    :param n_init: The number of initializations to perform. The best results are kept. default is 1
    :return: Dictionary: keys are clusters number. values are list of classifiers.
    '''
    gmm_clfs = {}
    for n in n_clusters:
        print(f'fitting {n} clusters with {iterations} iterations')
        clfs = []
        for _ in range(iterations):
            gmm=GaussianMixture(n, n_init=n_init, covariance_type='full').fit(X)
            clfs.append(gmm)
        gmm_clfs[n] = clfs
    return gmm_clfs


def get_bic_sils_scores(X, n_clusters, iterations, n_init=1):
    '''
    :param X: Data frame of the data
    :param gmm_clfs: Dictionary: keys are clusters number. values are list of classifiers.
    :param n_clusters: Range of clusters to fit
    :param iterations: Num of fits iteration in each cluster
    :param n_init: The number of initializations to perform. The best results are kept. default is 1
    :return:
    '''
    gmm_clfs = get_gmms_by_clusters(X=X, n_clusters=n_clusters, iterations=iterations, n_init=n_init)

    bics = []
    bics_err = []
    sils = []
    sils_err = []
    for n in n_clusters:
        tmp_bic = []
        tmp_sil = []
        for i in range(iterations):
            gmm = gmm_clfs[n][i]
            labels = gmm.predict(X)

            sil = silhouette_score(X, labels, metric='euclidean')
            bic = gmm.bic(X)

            tmp_bic.append(bic)
            tmp_sil.append(sil)
        bics.append(np.mean(tmp_bic))
        bics_err.append(np.std(tmp_bic))
        sils.append(np.mean(tmp_sil))
        sils_err.append(np.std(tmp_sil))
    return bics, bics_err, sils, sils_err


def get_curve_metrics(clf, X, y):
    '''
    :param clf: Classifier
    :param X: Data frame of the data
    :param y: The response feature
    :return: tuple(precisions, recalls, f1_scores, thresholds)
    Note: clf must implement 'score_samples' method.
    '''
    scores = clf.score_samples(X)
    precisions, recalls, thresholds = precision_recall_curve(y, -1 * scores)
    f1_scores = [2 * r * p / (r + p)
                 if r + p > 0
                 else 0
                 for r, p in zip(recalls, precisions)]
    return precisions, recalls, f1_scores, -thresholds


def f1_max_threshold(precisions, recalls, f1_scores, thresholds):
    '''
    :param precisions: list of possible precisions
    :param recalls: list of possible recalls
    :param f1_scores: list of possible f1 scores
    :param thresholds: list of all possible thresholds
    :return:
    '''
    t = thresholds[np.argmax(f1_scores)]
    p = precisions[np.argmax(f1_scores)]
    r = recalls[np.argmax(f1_scores)]
    f1 = np.max(f1_scores)

    return t, p, r, f1


def x_test_evalute_report(clf, X_test, y_test, threshold):
    '''
    :param clf: Classifier
    :param X_test: Data frame of the test data
    :param y_test: The response feature of the test data
    :param threshold: threshold for function score
    :return:
    '''
    scores = clf.score_samples(X_test)
    y_pred = scores.copy()
    y_pred[scores>threshold] = 0
    y_pred[scores<=threshold] = 1

    print(classification_report(y_true=y_test, y_pred=y_pred))