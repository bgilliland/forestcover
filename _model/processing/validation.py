from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from _model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    # new_vars_with_na = [
    #     var
    #     for var in config.model_settings.features
    #     if var
    #     not in config.model_settings.categorical_vars_with_na_frequent
    #     + config.model_settings.categorical_vars_with_na_missing
    #     + config.model_settings.numerical_vars_with_na
    #     and validated_data[var].isnull().sum() > 0
    # ]
    # validated_data.dropna(subset=new_vars_with_na, inplace=True)
    validated_data.dropna(inplace=True)

    return validated_data


def validate_inputs(
    *, input_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    relevant_data = input_data[config.model_settings.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=(
                validated_data.replace({np.nan: None}).to_dict(
                    orient="records"
                )
            )
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


# specify the input data schema. represents one record from the dataset.
class DataInputSchema(BaseModel):
    Elevation: Optional[float]
    Horizontal_Distance_To_Fire_Points: Optional[float]
    Horizontal_Distance_To_Roadways: Optional[float]
    Soil_Type: Optional[int]
    Horizontal_Distance_To_Hydrology: Optional[float]
    Vertical_Distance_To_Hydrology: Optional[float]
    Wilderness_Area: Optional[int]
    Aspect: Optional[float]
    Hillshade_Noon: Optional[float]
    Slope: Optional[float]


# second class represents the entire data set
# (a collection of DataInputSchema instances)
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
