import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pickle

def split_features_labels(df):
    features = df.loc[:, ~df.columns.str.contains('Damage')]
    labels = df.loc[:, df.columns.str.contains('Damage')]
    return features, labels

data = pd.read_csv('data/combined_clean_data.csv')
data = data.dropna()

machines = data['Machine'].unique()
separated_data = {'Bridgeport': data[data.Machine.str.contains('Bridgeport')], 'Lathe': data[data.Machine == 'Lathe 1'], 'Drill Press': data[data.Machine == 'Drill Press']}

models = {}
for machine in separated_data:
    df = separated_data[machine]
    df = df.drop(columns=['Time', 'Machine'])

    train, test = train_test_split(df, random_state=42)
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)

    y_train_y = np.zeros((y_train.shape[0], 1))
    y_train_x = np.zeros((y_train.shape[0], 1))
    y_train_y[y_train['Y_Damage Accumulation'] > 1.25] = 1
    y_train_x[y_train['X_Damage Accumulation'] > 1.25] = 1
    y_test_y = np.zeros((y_test.shape[0], 1))
    y_test_x = np.zeros((y_test.shape[0], 1))
    y_test_y[y_test['Y_Damage Accumulation'] > 1.25] = 1
    y_test_x[y_test['X_Damage Accumulation'] > 1.25] = 1


    clf_y = RandomForestClassifier(max_depth=11, random_state=42)
    clf_x = RandomForestClassifier(max_depth=11, random_state=42)

    clf_y.fit(X_train.values, y_train_y.reshape((-1,)))
    clf_x.fit(X_train.values, y_train_x.reshape((-1,)))

    y_pred_y = clf_y.predict(X_test.values)
    y_pred_x = clf_x.predict(X_test.values)

    y_f1 = metrics.f1_score(y_test_y, y_pred_y)
    x_f1 = metrics.f1_score(y_test_x, y_pred_x)

    # print('Most important Classifying feature:')
    # print('Machine: ', machine)
    # print('Y Damage:')
    # print(X_train.columns[np.argmax(clf_y.feature_importances_)])
    # print('X Damage:')
    # print(X_train.columns[np.argmax(clf_x.feature_importances_)])

    filename_y = ''.join(['models/', machine, '_clf_y', '.csv'])
    filename_x = ''.join(['models/', machine, '_clf_x', '.csv'])

    pickle.dump(clf_y, open(filename_y, 'wb'))
    pickle.dump(clf_x, open(filename_x, 'wb'))

    X_train_reg_y = X_train[y_train['Y_Damage Accumulation'] <= 1.25]
    X_train_reg_x = X_train[y_train['X_Damage Accumulation'] <= 1.25]
    y_train_reg_y = y_train['Y_Damage Accumulation'][y_train['Y_Damage Accumulation'] <= 1.25]
    y_train_reg_x = y_train['X_Damage Accumulation'][y_train['X_Damage Accumulation'] <= 1.25]

    X_test_reg_y = X_test[y_test['Y_Damage Accumulation'] <= 1.25]
    X_test_reg_x = X_test[y_test['X_Damage Accumulation'] <= 1.25]
    y_test_reg_y = y_test['Y_Damage Accumulation'][y_test['Y_Damage Accumulation'] <= 1.25]
    y_test_reg_x = y_test['X_Damage Accumulation'][y_test['X_Damage Accumulation'] <= 1.25]

    reg_x = RandomForestRegressor(max_depth=11, random_state=42)
    reg_y = RandomForestRegressor(max_depth=11, random_state=42)

    reg_x.fit(X_train_reg_x.values, y_train_reg_x.values.reshape((-1,)))
    reg_y.fit(X_train_reg_y.values, y_train_reg_y.values.reshape((-1,)))

    y_pred_x = clf_y.predict(X_train_reg_x.values)
    y_pred_y = clf_x.predict(X_train_reg_y.values)

    y_r2 = metrics.r2_score(y_train_reg_y, y_train_reg_y)
    x_r2 = metrics.r2_score(y_train_reg_x, y_train_reg_x)

    print(y_r2)
    print(x_r2)

