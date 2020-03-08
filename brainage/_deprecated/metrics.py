import torch


def mean_absolute_error(y, y_pred):
    '''
    Mean absolute error metric.
    Args:
        y:
        y_pred:

    Returns:

    '''
    absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
    _sum_of_absolute_errors = torch.sum(absolute_errors).item()
    _num_examples = y.shape[0]
    return _sum_of_absolute_errors / _num_examples


# todo add RSME