import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

SOURCE_PATH = Path("PEMS-BAY.csv")


class KNNImputerClass:
    def __init__(self, source_path: Path = SOURCE_PATH):
        self.source_path = source_path

    def define_noise_input(
        self,
        data: pd.DataFrame,
        ind2cols: dict[int, list[str]],
        use_positional_index: bool = False,
        copy: bool = True,
    ) -> pd.DataFrame:
        """Replace specified entries with NaN so KNNImputer treats them as missing."""
        df = data.copy() if copy else data

        for ind, cols in ind2cols.items():
            # Filter to existing columns to avoid KeyError
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols:
                continue

            if use_positional_index:
                row_label = df.index[ind]
                df.loc[row_label, valid_cols] = np.nan
            else:
                if ind in df.index:
                    df.loc[ind, valid_cols] = np.nan

        return df

    def _impute_missing_values(
        self, data: pd.DataFrame, n_neighbors: int = 5
    ) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit(data)
        imputed_data = imputer.transform(data)
        return pd.DataFrame(imputed_data, columns=data.columns, index=data.index)

    def perform_knn_imputation(
        self,
        data: pd.DataFrame,
        ind2cols: dict[int, list[str]],
        n_neighbors: int = 5,
        use_positional_index: bool = False,
    ) -> pd.DataFrame:
        noisy_data = self.define_noise_input(
            data, ind2cols, use_positional_index=use_positional_index, copy=True
        )
        return self._impute_missing_values(noisy_data, n_neighbors=n_neighbors)

