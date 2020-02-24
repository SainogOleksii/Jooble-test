import numpy as np
import pandas as pd

from functools import partial
from .utils import batch_generator, z_norm


class Kit2Preprocessor:
    def __init__(self, batch_size: int = 100, normalizer=z_norm):
        self.df = pd.DataFrame()
        self.batch_size = batch_size
        self.normalizer = normalizer

    def split_features(self):
        """
        Create a dataframe with columns for each feature.
        :return: self
        """
        result = pd.DataFrame()
        for batch in batch_generator(self.df, self.batch_size):
            feature_values = np.array(list(batch["features"].map(lambda x: np.int64(x.split(','))).values))[:, 1:]
            feature_names = [f"feature_2_{i}" for i in range(feature_values.shape[1])]
            new_data = pd.DataFrame(feature_values, columns=feature_names, index=batch.index)
            result = result.append(new_data)
        self.df = result
        return self

    def idx_max(self):
        """
        Find an index of max value for each row.
        :return: pd.Series of indexes.
        """
        result = pd.Series()
        for batch in batch_generator(self.df, self.batch_size):
            new_data = batch.idxmax(axis=1).str.split("_").map(lambda x: int(x[-1]))
            result = result.append(new_data)
        return result

    @staticmethod
    def get_abs(row: pd.Series, mean: pd.Series, max_index: pd.Series):
        """
        Calculate absolute difference between max_index value and mean value of max_index column for input vacancy.
        :param row: pd.Series of vacancy features.
        :param mean: pd.Series of mean values for vacancy features.
        :param max_index: pd.Series of max value indexes for each vacancy.
        :return: value of absolute difference for input vacancy.
        """
        feature_name = f"feature_2_{max_index.loc[row.name]}"
        return np.abs(row[feature_name] - mean[feature_name])

    def get_abs_mean_diff(self, mean: pd.Series = None, max_index: pd.Series = None):
        """
        Calculate absolute difference between max_index value and mean value of max_index column.
        :param mean: pd.Series of mean values for vacancy features.
        :param max_index: pd.Series of max value indexes for each vacancy.
        :return: pd.Series of absolute differences.
        """
        if mean is None:
            sum_ = np.zeros(self.df.shape[1])
            count = 0
            for batch in batch_generator(self.df, self.batch_size):
                sum_ += batch.sum()
                count += batch.shape[0]
            mean = sum_ / count
        if max_index is None:
            max_index = self.idx_max()
        result = pd.Series()
        for batch in batch_generator(self.df, self.batch_size):
            new_data = batch.apply(partial(self.get_abs, mean=mean, max_index=max_index), axis=1)
            result = result.append(new_data)
        return result

    def preprocess(self, data_frame: pd.DataFrame):
        """
        Preprocess input dataframe
        :param data_frame: pd.DataFrame vacancies.
        :return: pd.DataFrame of preprocess vacancies features.
        """
        self.df = data_frame
        self.split_features()
        max_index = self.idx_max()
        res_df, (mean, _) = self.normalizer(self.df, self.batch_size)
        abs_mean_diff = self.get_abs_mean_diff(mean=mean, max_index=max_index)
        res_df["max_feature_2_index"] = max_index
        res_df["max_feature_2_abs_mean_diff"] = abs_mean_diff
        self.df = res_df
        return self.df
