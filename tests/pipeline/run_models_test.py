import pytest

from pipeline.run_models import save_model_hyperparams_metrics, save_model


@pytest.mark.parametrize(
    "model_name, prefix",
    [
        ("rf_model", "010230_"),
        ("mlp_model", "010230_"),
        ("logreg", "010230_"),
        ("xgb", "010230_"),
    ],
)
def test_save_model(model_name, prefix, mocker):
    # Mock the model and joblib_dump function
    model = mocker.Mock()
    joblib_dump_mock = mocker.Mock()
    mocker.patch("joblib.dump", joblib_dump_mock)

    save_model(model_name, model, prefix)

    expected_call = mocker.call(model, f"{prefix}{model_name}.joblib")
    assert joblib_dump_mock.called_once_with(model, expected_call)


@pytest.mark.parametrize(
    "model_name, params, prefix",
    [
        ("rf_model", {"param1": 1, "param2": 2, "param3": 3}, "010230_"),
        ("mlp_model", {"param1": 3, "param2": 4, "param7": 6}, "010233_"),
        ("logreg", {"param1": -1, "param6": 2}, "010230_"),
        ("xgb", {"param2": 2, "param4": 3}, "011230_"),
    ],
)
def test_save_params(model_name, params, prefix, mocker):
    # Mock the open function and the file object
    file_mock = mocker.Mock()
    open_mock = mocker.Mock(return_value=file_mock)
    open_mock.__enter__ = mocker.Mock(return_value=file_mock)
    open_mock.__exit__ = mocker.Mock()

    with mocker.patch("builtins.open", open_mock):
        # Call the save_params function
        save_model_hyperparams_metrics(model_name, mocker.Mock(), params, mocker.Mock(), prefix)

        # Check that the file was opened and written to correctly
        expected_file = prefix + "params"
        expected_write = f"\nmodel: {model_name}, params: {params}"
        open_mock.assert_called_once_with(expected_file, "a")
        file_mock.write.assert_called_once_with(expected_write)


@pytest.mark.parametrize(
    "model_name, metrics, prefix",
    [
        ("rf_model", {"mae": 1.3, "mape": 1.3, "rmse": 3.3}, "010230_"),
        ("mlp_model", {"mae": 3.1, "mape": 4.7, "rmse": 6.0}, "010233_"),
        ("logreg", {"mae": 0.43, "mape": 5.05, "rmse": 3.5}, "010230_"),
        ("xgb", {"mae": 2.4, "mape": 3.5, "rmse": 6.4}, "011230_"),
    ],
)
def test_save_metrics(model_name, metrics, prefix, mocker):
    # Mock the open function and the file object
    file_mock = mocker.Mock()
    open_mock = mocker.Mock(return_value=file_mock)
    open_mock.__enter__ = mocker.Mock(return_value=file_mock)
    open_mock.__exit__ = mocker.Mock()

    with mocker.patch("builtins.open", open_mock):
        # Call the save_params function
        save_model_hyperparams_metrics(model_name, mocker.Mock(), mocker.Mocker(), metrics, prefix)

        # Check that the file was opened and written to correctly
        expected_file = prefix + "params"
        expected_metrics_values = [f", {val}" for val in metrics.values()]
        expected_write = f"{model_name}{expected_metrics_values}"
        open_mock.assert_called_once_with(expected_file, "a")
        file_mock.write.assert_called_once_with(expected_write)
