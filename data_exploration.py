import pandas as pd
from visualizations import plot_class_dist, plot_time_amount_dist


DATA_SET_PATH = 'dataset/creditcard.csv'
# load the data set
df = pd.read_csv(DATA_SET_PATH)

if __name__ == '__main__':
    print('Data set 5 first rows')
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
        print(df.head())
    print()
    print('Data set columns:')
    print(df.columns)

    print()
    print('Data set columns statistics:')
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
        print(df.describe())

    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_values_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
    print()
    print('Data set missing values:')
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
        print(missing_values_df)

    plot_class_dist(df=df)
    plot_time_amount_dist(df=df)

