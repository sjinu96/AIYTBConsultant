from sklearn import utils

import pandas as pd
from preprocessing import Preprocessor


class DataLoader(Preprocessor):
    def __init__(self, path):
        self._raw = pd.read_csv(path)
        self.data = super()._fit_transform(self._raw.copy())
        print("Data Loaded. :P")

    def get_data(
        self,
        filter_size=7,
        target_size=1,
        stride=1,
        drop_features=None,
        target_features=None,
        channel: list = None,
        shuffle=False,
        random_state=None,
        order=None,
    ):
        if channel == None:
            data_to_extract = self.data
        else:
            channels = self.list_channel[channel].tolist()
            data_to_extract = self.data.set_index("channel").loc[channels].reset_index()

        train, target = self._create_sequential_data(
            data_to_extract,
            filter_size,
            target_size,
            stride,
            drop_features,
            target_features,
        )
        if shuffle:
            train, target = utils.shuffle(train, target, random_state=random_state)

        if order:
            train_col = [col for col in train.columns.unique().tolist()]
            target_col = [col for col in target.columns.unique().tolist()]
            train = train[train_col]
            target = target[target_col]

        return train, target

    @property
    def list_features(self):
        # list the entire features, hence you can choose which features are included in whole set.
        return self.data.columns.tolist()

    @property
    def list_channel(self):
        # list indices of channel.
        return pd.Series(self.data.channel.unique().tolist())
