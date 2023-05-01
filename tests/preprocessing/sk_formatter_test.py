from pipeline.preprocessing.sk_formatter import SKFormatter
from tests.mock_dataset import mock_dataset


def test_encode_single_value_features():
    skf = SKFormatter(None)
