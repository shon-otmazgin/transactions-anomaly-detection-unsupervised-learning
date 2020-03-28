from warnings import filterwarnings
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from utils import split_train_test
from visualizations import plot_bic_slis_scores
filterwarnings('ignore')

RANDOM_STATE = 42
N_CLUSTERS = np.arange(2, 11)
ITERATIONS = 20
N_INIT = 2
DATA_SET_PATH = 'dataset/creditcard.csv'

# loading  the data set
df = pd.read_csv(DATA_SET_PATH)


def get_gmms_by_clusters(X, n_clusters, iterations, n_init=1):
    '''
    :param X: Data set
    :param n_clusters: Range of clusters to fit
    :param iterations: Num of fits iteration in each cluster
    :param n_init: The number of initializations to perform. The best results are kept. default is 1
    :return: Dictionary: keys are clusters number. values are list of classifiers.
    '''
    gmm_clfs = {} # dict for clf with n componnents and clf iterations
    for n in n_clusters:
        print(f'fitting {n} clusters with {iterations} iterations')
        clfs = []
        for _ in range(iterations):
            gmm=GaussianMixture(n, n_init=n_init, covariance_type='full').fit(X)
            clfs.append(gmm)
        gmm_clfs[n] = clfs
    return gmm_clfs


def get_bic_sils_scores(gmm_clfs, n_clusters, iterations):
    '''
    :param gmm_clfs: Dictionary: keys are clusters number. values are list of classifiers.
    :param n_clusters: Range of clusters to fit
    :param iterations: Num of fits iteration in each cluster
    :return:
    '''
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


if __name__ == '__main__':
    X, X_test, y, y_test = split_train_test(df=df)

    gmm_clfs = get_gmms_by_clusters(X=X, n_clusters=N_CLUSTERS, iterations=ITERATIONS, n_init=N_INIT)
    print()
    print(f'Calculte BIC and Silhouette Score for {N_CLUSTERS} clusters')
    bics, bics_err, sils, sils_err = get_bic_sils_scores(gmm_clfs=gmm_clfs,
                                                         n_clusters=N_CLUSTERS,
                                                         iterations=ITERATIONS)
    print()
    print('Plotting scores...')
    plot_bic_slis_scores(n_clusters=N_CLUSTERS, bics=bics, bics_err=bics_err, sils=sils, sils_err=sils_err)

