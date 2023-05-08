import pytest
from pipeline.run_models import save_metrics

@pytest.mark.parametrize(
    "input_dict, path",
    [
        (
            { "MLP": {"mae": 0.5, "mape": 0.4},
              "RF": {"mae": 0.6, "mape": 0.7}
            },
            "./"
        )
    ]
)

def test_save_metrics(input_dict, path):

    #save_metrics(input_dict, path)
    pass
