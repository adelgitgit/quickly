import pandas as pd


class PemsRowSmoothing:

    def __init__(self, data, k=3):
        self.data = data.copy()

    def transform(self,k=3):
        smoothed = (
            self.data.T
            .rolling(window=k, min_periods=1)
            .mean()
            .T
        )

        return smoothed


