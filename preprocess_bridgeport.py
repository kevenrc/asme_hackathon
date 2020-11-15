import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob

class DataLoader():

    def __init__(self, filename):
        self.data = pd.read_csv(filename).dropna(axis=1, how='all')
        self.clean_data()
        self.split_features_labels()
        # self.normalize_values()

    def clean_data(self):
        data_columns = self.data.columns.to_series()
        data_columns = data_columns[~data_columns.str.contains('Unnamed')]
        self.data.columns = pd.MultiIndex.from_product([data_columns.values, ['Time', 'Value']])
        self.data = self.data.drop(0).reset_index(drop=True)
        parent_columns = pd.Series([a[0] for a in self.data.columns])
        df = pd.DataFrame({'Time': pd.Series([], dtype='datetime64[ns]')})
        for i, pc in enumerate(parent_columns):
            if i == 0:
                df.Time = pd.to_datetime(self.data[pc].Time)
                df = df.dropna(axis=0, how='any')
            if i % 2 == 0:
                df2 = self.data[pc]
                df2.columns = ['Time', pc]
                df2.Time = pd.to_datetime(df2.Time)
                df2 = df2.dropna(axis=0, how='any')
                df = pd.merge_asof(df, df2, on='Time')
        self.data = df.sort_values(by='Time')

    def split_features_labels(self):
        parent_columns = self.data.columns
        self.data_features = self.data[self.data.columns[~parent_columns.str.contains('Damage')]]
        self.data_labels = self.data[self.data.columns[parent_columns.str.contains('Damage')]]

    def normalize_values(self):
        idx = pd.IndexSlice
        self.feature_values = self.data_features.loc[:, idx[:, 'Value']].apply(pd.to_numeric)
        self.label_values = self.data_labels.loc[:, idx[:, 'Value']].apply(pd.to_numeric)

        self.feature_values = (self.feature_values - self.feature_values.min()) / (self.feature_values.max() - self.feature_values.min())

    def plot_distributions(self):
        self.feature_values.plot.kde(subplots=True, layout=(4, 3))
        plt.show()

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
     

if __name__ == "__main__":
    filenames = glob.glob('data/*test.csv')
    for filename in filenames:
        print(filename)
        data = DataLoader(filename)
        print(data.data.shape)
        data.data.to_csv(''.join([filename.replace('.csv', ''), '_clean.csv']), index=False)