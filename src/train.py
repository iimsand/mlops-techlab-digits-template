import argparse
from sklearn import svm
import yaml
from typing import Text
import joblib
from pathlib import Path
import pandas as pd

from src.utils import get_logger, use_file


def train_model(config_path: Text) -> any:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('train_model', log_level=config['base']['log_level'])

    logger.info('Load train dataset')
    train_df = pd.read_csv(config["data"]["train_path"])
    X_train = train_df.drop('target', axis=1).values.astype('float32')
    y_train = train_df.loc[:, 'target'].values.astype('int32')

    logger.info('Train model')
    model = svm.SVC(kernel=config["train"]["svc_params"]
                    ["kernel"], C=config["train"]["svc_params"]["C"])
    model.fit(X_train, y_train)

    logger.info('Save model')
    joblib.dump(model, use_file(config["train"]["model_path"]))

    return model


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
