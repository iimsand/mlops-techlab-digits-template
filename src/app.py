import argparse
from typing import Text
from src.utils import get_logger
import gradio as gr
import yaml
import joblib
from skimage.transform import resize
import numpy as np

np.set_printoptions(suppress=True)

def app(config_path: Text) -> any:
    """A sample application to test a trained model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger(
        'app', log_level=config['base']['log_level'])

    logger.info('Load model')
    model = joblib.load(config["train"]["model_path"])

    def preprocess(data):
        img = resize(data, (8, 8))
        return [img.reshape(64)]

    def predict(data):
        if (data is not None):
            input_data = preprocess(data)
            return int(model.predict(input_data)[0])
        else:
            return "No data"

    gr.Interface(fn=predict,
                 inputs='sketchpad',
                 outputs='label',
                 live=True,
                 analytics_enabled=False,
                 allow_flagging=False).launch(share=True)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config',
                             required=False, default='params.yaml')
    args = args_parser.parse_args()

    app(config_path=args.config)
