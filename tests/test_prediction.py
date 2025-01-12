import numpy as np

from _model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    # expected_first_prediction_value = 113422
    expected_no_predictions = 116203

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    # print(predictions)
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.integer)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    # assert predictions[0] == expected_first_prediction_value
