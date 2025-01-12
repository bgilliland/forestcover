import numpy as np

from _model.config.core import config
from _model.processing.features import TrigTransformer


def test_trig_var_bounds(sample_input_data):
    # then
    assert max(sample_input_data["Aspect"]) <= 360
    assert min(sample_input_data["Aspect"]) >= 0
    assert max(sample_input_data["Slope"]) <= 360
    assert min(sample_input_data["Slope"]) >= 0


def test_trig_transformer(sample_input_data):
    # Given
    transformer = TrigTransformer(
        variables=config.model_settings.trig_transform,  # YearRemodAdd
    )
    assert sample_input_data["Aspect"].iat[0] == 232

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["SinAspect"].iat[0] == np.sin(
        np.radians(sample_input_data["Aspect"].iat[0])
    )
