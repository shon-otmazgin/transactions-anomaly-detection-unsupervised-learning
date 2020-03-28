import pandas as pd
from utils.visualizations import plot_class_dist, plot_time_amount_dist


DATA_SET_PATH = 'dataset/creditcard.csv'
# loading  the data set
df = pd.read_csv(DATA_SET_PATH)

if __name__ == '__main__':
    print('Data set 5 first rows')
    pd.set_option('float_format', '{:f}'.format)
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False, 'float_format', '{:.3f}'.format):
        print(df.head())
    print()
    print('Data set columns:')
    print(df.columns)

    print()
    print('Data set columns statistics:')
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False, 'float_format', '{:.2f}'.format):
        print(df.describe())

    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_values_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
    print()
    print('Data set missing values:')
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False, 'float_format', '{:.1f}'.format):
        print(missing_values_df)

    print()
    print('Plotting 2 figures ...')
    print('1. Class distributions')
    plot_class_dist(df=df)
    print('2. Time and Amount distributions')
    plot_time_amount_dist(df=df)

