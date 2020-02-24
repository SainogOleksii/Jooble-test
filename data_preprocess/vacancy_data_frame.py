import numpy as np
import pandas as pd

from .utils import batch_generator


class VacancyDataFrame:
    def __init__(self, data_frame: pd.DataFrame, batch_size: int = 10):
        self.df = data_frame
        self.batch_size = batch_size
        self.kits = {}

    def split_into_kits(self):
        """
        Create a dictionary of kits with different features.
        :return: self
        """
        for batch in batch_generator(self.df, self.batch_size):
            batch.loc[:, "kit_id"] = batch["features"].str.split(",").map(lambda x: np.int64(x[0]))
            for kit_id in batch["kit_id"].unique():
                if kit_id not in self.kits.keys():
                    self.kits.update({kit_id: pd.DataFrame()})
                new_data = self.kits.get(kit_id).append(batch[batch["kit_id"] == kit_id])
                self.kits.update({kit_id: new_data})
        return self

    def preprocess_kit(self, preprocessor, kit_id: int = 2):
        """
        Preprocess kit with input kit_id.
        :param preprocessor: kit preprocessor object.
        :param kit_id: id of kit.
        :return: self
        """
        df = self.kits.get(kit_id)
        self.kits.update({kit_id: preprocessor.preprocess(df)})
        return self

    def get_kit(self, kit_id):
        """
        Return pd.DataFrame of input kit_id.
        :param kit_id: id of kit.
        :return: pd.DataFrame
        """
        return self.kits.get(kit_id)
