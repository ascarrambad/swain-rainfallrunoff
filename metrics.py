
import torch
from torchmetrics.utilities.checks import _check_same_shape

from tsl.nn.metrics.metric_base import MaskedMetric


def _nse(y_hat, y):
    denominator = ((y - y.mean())**2).sum()
    numerator = ((y_hat - y)**2).sum()

    return 1 - numerator / denominator

class MaskedNSE(MaskedMetric):
    """
        Nash-Sutcliffe efficiency coefficient.

        Args:
            mask_nans (bool, optional): Whether to automatically mask nan values.
            mask_inf (bool, optional): Whether to automatically mask infinite values.
            compute_on_step (bool, optional): Whether to compute the metric right-away or if accumulate the results.
                             This should be `True` when using the metric to compute a loss function, `False` if the metric
                             is used for logging the aggregate error across different minibatches.
            at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):

        super(MaskedNSE, self).__init__(metric_fn=_nse,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs=None,
                                        at=at)


    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        mask = self._check_mask(mask, y)
        y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
        y = torch.where(mask, y, torch.zeros_like(y))
        val = self.metric_fn(y_hat, y)
        return val.sum(), mask.sum()


################################################################################

def np_masked_nse(y_hat, y, mask=None):
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    return _nse(y_hat[mask], y[mask])