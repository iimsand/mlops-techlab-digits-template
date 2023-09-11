import argparse
from sklearn import datasets, utils
import yaml
from typing import Text
import joblib

from src.utils import get_logger, use_file


def data_load(config_path: Text) -> utils.Bunch:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('data_load', log_level=config['base']['log_level'])

    logger.info('Get dataset')
    digits = datasets.load_digits()

    logger.info('Save raw data')
    joblib.dump(digits, use_file(config["data"]["dataset_joblib"]))

    return digits


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
