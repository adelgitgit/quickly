
import pandas as pd
import numpy as np


class TimeFeatureEngineer:

    def __init__(self):
        pass

    def normalize_time(self,data):
        """
        Normalize hour of day into [0, 1).
        """
        hours = (
            data.index.hour
            + data.index.minute / 60
        )

        data["time_normalized"] = hours / 24

        return data

    def cyclical_time(self, data):
        """
        Encode time using sine and cosine.
        """
        hours = (
            data.index.hour
            + data.index.minute / 60
        )

        data["time_sin"] = np.sin(
            2 * np.pi * hours / 24
        )

        data["time_cos"] = np.cos(
            2 * np.pi * hours / 24
        )

        return data

    def transform(self,data, method="cyclical"):

        if method == "normalized":
            return self.normalize_time(data)

        elif method == "cyclical":
            return self.cyclical_time(data)

        else:
            raise ValueError(
                "method must be 'normalized' or 'cyclical'"
            )

