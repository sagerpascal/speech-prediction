from pytorch_lightning.metrics.regression import ExplainedVariance, MeanSquaredError, MeanAbsoluteError


class ExplainedVarianceUA(ExplainedVariance):
    __name__ = 'Explained Variance Uniform Average'

    def __init__(self):
        super().__init__(multioutput='uniform_average')


class ExplainedVarianceVW(ExplainedVariance):
    __name__ = 'Explained Variance Variance Weighted'

    def __init__(self):
        super().__init__(multioutput='variance_weighted')


class MSE(MeanSquaredError):
    __name__ = 'Mean Squared Error'


class MAE(MeanAbsoluteError):
    __name__ = 'Mean Absolute Error'


def get_metrics(conf):
    metrics = [ExplainedVarianceUA(),
               ExplainedVarianceVW(),
               MSE(),
               MAE(),
               ]

    for m in metrics:
        m.to(conf['device'])
    return metrics
