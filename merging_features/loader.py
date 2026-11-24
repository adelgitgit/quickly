import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PEMSForecastDataset(Dataset):

    def __init__(
        self,
        original,
        smoothed,
        seq_len=12,
        pred_len=12
    ):

        # Validate DataFrame inputs
        if isinstance(original, pd.DataFrame):

            assert original.columns.equals(smoothed.columns), \
                "Original and smoothed columns must match."

            assert original.index.equals(smoothed.index), \
                "Original and smoothed indexes must match."

            assert original.index.is_monotonic_increasing, \
                "Data must be chronologically sorted."

        # Convert to tensors
        self.original = torch.tensor(
        original.to_numpy(copy=True) if isinstance(original, pd.DataFrame)
        else np.array(original, copy=True),
        dtype=torch.float32
        )
        self.smoothed = torch.tensor(
        smoothed.to_numpy(copy=True) if isinstance(smoothed, pd.DataFrame)
        else np.array(smoothed, copy=True),
        dtype=torch.float32
    )

        self.seq_len = seq_len
        self.pred_len = pred_len

        # Validate dimensions
        assert self.original.shape == self.smoothed.shape

        assert self.original.ndim == 2, \
            "Expected data shape: (time_steps, num_sensors)"

        assert seq_len > 0 and pred_len > 0

        assert len(self.original) >= seq_len + pred_len

        assert torch.isfinite(self.original).all(), \
            "Original data contains NaN or Inf."

        assert torch.isfinite(self.smoothed).all(), \
            "Smoothed data contains NaN or Inf."

    def __len__(self):

        return len(self.original) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):

        # Historical input
        X1 = self.smoothed[
            idx: idx + self.seq_len
        ]

        X2 = self.original[
            idx: idx + self.seq_len
        ]

        # Future target
        Y = self.original[
            idx + self.seq_len:
            idx + self.seq_len + self.pred_len
        ]

        return X1, X2, Y