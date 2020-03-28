from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
RANDOM_STATE = 42


def __scale_dataset(df):
    '''
    :param df: Credit card data frame
    :return: scaled data frame for columns Amount and Time with Robust Robust Scaler
    '''
    scaled_df = df.drop(columns=['Class'])
    scaled_df['Amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    scaled_df['Time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    scaled_df['Class'] = df['Class']
    return scaled_df


def split_train_test(df):
    '''
    :param df: Credit card data frame
    :return: X, X_test, y, y_test X is 80% of the data.
    '''
    scaled_df = scaled_df = __scale_dataset(df=df)
    train_df, test_df = train_test_split(scaled_df, test_size=0.2, random_state=RANDOM_STATE)

    # This is just to save time during training.
    train_df = train_df.sample(frac=0.2, random_state=RANDOM_STATE)
    test_df = test_df.sample(frac=0.2, random_state=RANDOM_STATE)

    X = train_df.drop(columns=['Class'], axis=1).reset_index(drop=True)
    y = train_df['Class'].reset_index(drop=True)
    X_test = test_df.drop(columns=['Class'], axis=1).reset_index(drop=True)
    y_test = test_df['Class'].reset_index(drop=True)

    print(f'Train size: {X.shape}')
    print(f'Test size: {X_test.shape}')
    print()
    return X, X_test, y, y_test