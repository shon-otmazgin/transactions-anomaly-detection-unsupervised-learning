import pandas as pd
import numpy as np
from utils.scale_and_split import split_train_test
from utils.metrics import get_bic_sils_scores
from utils.visualizations import plot_bic_slis_scores


N_CLUSTERS = np.arange(2, 11)
ITERATIONS = 20
N_INIT = 2
DATA_SET_PATH = 'dataset/creditcard.csv'

# loading  the data set
df = pd.read_csv(DATA_SET_PATH)


if __name__ == '__main__':
    X, X_test, y, y_test = split_train_test(df=df)

    print('Note: Running BIC and Silhouette Scores on the data could take 3 hours!')
    print(f'Calculte BIC and Silhouette Scores for {N_CLUSTERS} clusters')
    bics, bics_err, sils, sils_err = get_bic_sils_scores(X=X,
                                                         n_clusters=N_CLUSTERS,
                                                         iterations=ITERATIONS,
                                                         n_init=N_INIT)
    print()
    print('Plotting scores...')
    plot_bic_slis_scores(n_clusters=N_CLUSTERS, bics=bics, bics_err=bics_err, sils=sils, sils_err=sils_err)

