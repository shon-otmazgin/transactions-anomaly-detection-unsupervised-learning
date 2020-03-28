import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import get_curve_metrics, f1_max_threshold
from sklearn.metrics import auc


def plot_class_dist(df):
    ax = df['Class'].value_counts(normalize=True).plot(kind='bar', figsize=(7, 4))
    for i, v in enumerate(df['Class'].value_counts(normalize=True).values):
        ax.text(i - 0.1, v + 0.005, str(round(v, 4)))
    ax.set_xticklabels(['Valid', 'Fruad'], rotation=0)
    ax.set_title('Class Distribution', fontsize=14)

    fig = plt.gcf()
    fig.canvas.set_window_title('Class Distribution')
    plt.show()


def plot_time_amount_dist(df):
    fig, ax = plt.subplots(3, 2, figsize=(14, 8))
    plt.subplots_adjust(hspace=0.6)

    amount_val = df['Amount'].values
    time_val = df['Time'].values

    sns.distplot(amount_val, ax=ax[0, 0], color='r')
    ax[0, 0].set_title('Distribution of Transaction Amount', fontsize=14)
    ax[0, 0].set_xlim([min(amount_val), max(amount_val)])

    sns.distplot(time_val, ax=ax[0, 1], color='b')
    ax[0, 1].set_title('Distribution of Transaction Time', fontsize=14)
    ax[0, 1].set_xlim([min(time_val), max(time_val)])

    time_class_0 = df.loc[df['Class'] == 0]["Time"]
    time_class_1 = df.loc[df['Class'] == 1]["Time"]
    amount_class_0 = df.loc[df['Class'] == 0]["Amount"]
    amount_class_1 = df.loc[df['Class'] == 1]["Amount"]

    sns.distplot(amount_class_0, ax=ax[1, 0], color='b', hist=True, label='Valid')
    sns.distplot(amount_class_1, ax=ax[1, 0], color='r', hist=True, label='Fraud')
    ax[1, 0].set_title('Distribution of Transaction Amount', fontsize=14)
    ax[1, 0].legend()

    sns.distplot(time_class_0, ax=ax[1, 1], color='b', hist=True, label='Valid')
    sns.distplot(time_class_1, ax=ax[1, 1], color='r', hist=True, label='Fraud', bins=25)
    ax[1, 1].set_title('Distribution of Transaction Time', fontsize=14)
    ax[1, 1].legend()

    ax[2, 0].scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
    ax[2, 0].set_title('Fraud', fontsize=14)
    ax[2, 0].set_ylim([min(amount_val), max(amount_val)])
    ax[2, 0].set_ylabel('Amount')
    ax[2, 0].set_xlabel('Time (in Seconds)')

    ax[2, 1].scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
    ax[2, 1].set_title('Valid', fontsize=14)
    ax[2, 1].set_ylim([min(amount_val), max(amount_val)])
    ax[2, 1].set_ylabel('Amount')
    ax[2, 1].set_xlabel('Time (in Seconds)')

    fig = plt.gcf()
    fig.canvas.set_window_title('Time and  Amount Distributions')
    plt.show()


def plot_bic_slis_scores(n_clusters, bics, bics_err, sils, sils_err):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.errorbar(n_clusters, bics, yerr=bics_err, label='BIC')
    ax1.set_title("BIC Scores", fontsize=14)
    ax1.set_ylabel("Score")
    ax1.set_xlabel("N. of clusters")
    ax1.legend()

    ax2.errorbar(n_clusters, sils, yerr=sils_err, label='Silohuette')
    ax2.set_title("Silhouette Scores", fontsize=14)
    ax2.set_ylabel("Score")
    ax2.set_xlabel("N. of clusters")
    ax2.legend()

    fig = plt.gcf()
    fig.canvas.set_window_title('BIC and Silohuette scores')
    plt.show()


def plot_recall_precision_curve_samples_scores(clf, X, y, clf_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    precisions, recalls, f1_scores, thresholds = get_curve_metrics(clf, X, y)
    t, p, r, f1 = f1_max_threshold(precisions, recalls, f1_scores, thresholds)
    area = auc(recalls, precisions)

    ax1.plot(recalls, precisions, marker=',', label=clf_name)
    ax1.axvline(r, linestyle='dashed', color='c')
    ax1.axhline(p, linestyle='dashed', color='c')
    ax1.set_title(f'Precision Recall Curve AUC: {area:.3f}', fontsize=14)
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')
    ax1.text(r + 0.01, p + 0.01, f'F1={str(round(f1, 3))}\nT={t:.3f}')
    ax1.legend(loc='upper center')

    scores = clf.score_samples(X)

    frauds = scores[y[y == 1].index]
    frauds_indices = y[y == 1].index
    valid = scores[y[y == 0].index]
    valid_indices = y[y == 0].index

    ax2.scatter(valid_indices, valid, cmap='coolwarm', label=f'Valid {len(y[y == 0])}')
    ax2.scatter(frauds_indices, frauds, cmap='coolwarm', label=f'Frauds {len(y[y == 1])}')
    ax2.legend()
    ax2.axhline(t, linestyle='dashed', color='red')
    ax2.set_title('Score function', fontsize=14)
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Sample')

    fig = plt.gcf()
    fig.canvas.set_window_title('precision_recall_sample_scores')

    plt.show()


def plot_recall_precision_change(recalls, precisions):
    fig, ax1 = plt.subplots(1, 1, figsize=(7,4))

    ax1.plot(recalls, label='Recall')
    ax1.plot(precisions, label='Precision')
    ax1.set_title('Recall-Precision Change', fontsize=14)
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Iteration')
    ax1.set_xticks(range(0,len(recalls), 2))
    ax1.legend()

    fig = plt.gcf()
    fig.canvas.set_window_title('precision_recall_change')

    plt.show()


