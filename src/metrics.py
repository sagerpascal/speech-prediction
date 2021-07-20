from pytorch_lightning.metrics.regression import MeanSquaredError, MeanAbsoluteError


class MSE(MeanSquaredError):
    __name__ = 'Mean Squared Error'


class MAE(MeanAbsoluteError):
    __name__ = 'Mean Absolute Error'


def get_metrics(conf, device):
    """ Returns the metrics used to evaluate the results """
    metrics = [
        MSE(),
        MAE(),
    ]

    for m in metrics:
        m.to(device)
    return metrics
