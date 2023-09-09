import argparse
import yaml
from typing import Text
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger, use_file


def data_split(config_path: Text) -> (pd.DataFrame, pd.DataFrame):
    """Split dataset.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('data_split', log_level=config['base']['log_level'])

    logger.info('Load features')
    data = pd.read_csv(config["data"]["features_path"])

    logger.info('Split data')
    train_dataset, test_dataset = train_test_split(
        data, test_size=config["data"]["test_size"], shuffle=False)

    logger.info('Save split data')
    train_dataset.to_csv(use_file(config["data"]["train_path"]), index=False)
    test_dataset.to_csv(use_file(config["data"]["test_path"]), index=False)

    return train_dataset, test_dataset


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)
