import numpy as np
import pandas as pd


def batch_generator(data_frame: pd.DataFrame, batch_size: int):
    """
    Generator of batches.
    :param data_frame: pd.DataFrame of input data.
    :param batch_size: batch size.
    :return: batches of input data.
    """
    while data_frame.shape[0]:
        yield data_frame.iloc[:batch_size]
        data_frame = data_frame.iloc[batch_size:]


def z_norm(data_frame: pd.DataFrame, batch_size: int = 100):
    """
    Z-normalization of input data.
    :param data_frame: pd.DataFrame of input data.
    :param batch_size: batch size.
    :return: Normalize pd.DataFrame.
    """
    sum_ = np.zeros(data_frame.shape[1])
    square = np.zeros(data_frame.shape[1])
    count = 0
    for batch in batch_generator(data_frame, batch_size):
        sum_ += batch.sum()
        count += batch.shape[0]
        square += (batch ** 2).sum()
    mean = sum_ / count
    std = (square / (count - 1) - count / (count - 1) * mean ** 2) ** 0.5

    result = pd.DataFrame()
    for batch in batch_generator(data_frame, batch_size):
        new_data = (batch - mean) / std
        new_name = (lambda x: "_".join(x.split("_")[:-1] + ["stand", x.split('_')[-1]]))
        new_data.columns = [new_name(column) for column in new_data.columns]
        result = result.append(new_data)
    return result, (mean, std)
