import argparse
from sklearn import metrics
import yaml
from typing import Text
import joblib
import pandas as pd
import json

from src.utils import get_logger, use_file


def evaluate_model(config_path: Text) -> any:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger(
        'evaluate_model', log_level=config['base']['log_level'])

    logger.info('Load test dataset')
    test_df = pd.read_csv(config["data"]["test_path"])
    X_test = test_df.drop('target', axis=1).values.astype('float32')
    y_test = test_df.loc[:, 'target'].values.astype('int32')

    logger.info('Load model')
    model = joblib.load(config["train"]["model_path"])

    logger.info('Test model')
    y_pred = model.predict(X_test)

    logger.info('Calculate f1 score')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')

    logger.info('Save metrics')
    json_metrics = {'f1': f1}
    with open(use_file(config["reports"]["metrics_path"]), 'w') as metrics_file:
        json.dump(obj=json_metrics, fp=metrics_file, indent=4)

    logger.info('Plot confusion matrix')
    cm_plot = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    logger.info('Save confusion matrix')
    cm_plot.figure_.savefig(
        use_file(config["reports"]["cm_plot_path"]))

    return json_metrics, cm_plot


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
