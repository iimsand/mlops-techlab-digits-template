import argparse
import yaml
from typing import Text
import joblib
import numpy as np
import pandas as pd

from src.utils import get_logger, use_file


def data_prepare(config_path: Text) -> np.ndarray:
    """Prepare the data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('data_prepare', log_level=config['base']['log_level'])

    logger.info('Load raw data')
    dataset = joblib.load(config["data"]["dataset_joblib"])

    logger.info('Prepare data')
    prepared_dataset = pd.DataFrame(
        data=np.c_[dataset.data, dataset.target], columns=dataset.feature_names + ['target'])

    logger.info('Save prepared data')
    prepared_dataset.to_csv(
        use_file(config["data"]["features_path"]), index=False)

    return prepared_dataset


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_prepare(config_path=args.config)
