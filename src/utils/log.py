import logging
import datetime
import sys
import wandb


def setup_logging():
    logging.basicConfig(filename="logs/{}.log".format(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")), filemode="w", level=logging.DEBUG)

    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


def setup_wandb():
    return wandb.init(project="ASR temporal context", entity="ASR", job_type='train')

