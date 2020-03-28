import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_dist(df):
    ax = df['Class'].value_counts(normalize=True).plot(kind='bar', figsize=(7, 4))
    for i, v in enumerate(df['Class'].value_counts(normalize=True).values):
        ax.text(i - 0.1, v + 0.005, str(round(v, 4)))
    ax.set_xticklabels(['Valid', 'Fruad'], rotation=0)
    ax.set_title('Class Distribution', fontsize=14)

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

    plt.show()