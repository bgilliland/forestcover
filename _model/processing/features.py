from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from _model.config.core import config


class TrigTransformer(BaseEstimator, TransformerMixin):
    """Transform degree to sin"""

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[f"Sin{feature}"] = np.sin(np.radians(X[feature]))
            X[f"Cos{feature}"] = np.cos(np.radians(X[feature]))
        to_drop = (
            self.variables + config.model_settings.transformed_features_drop
        )
        print(f"dropping {to_drop}")
        X.drop(
            columns=to_drop,
            inplace=True,
            errors="ignore",
        )

        return X
