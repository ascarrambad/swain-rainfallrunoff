
import numpy as np
import torch
from torchmetrics.utilities.checks import _check_same_shape

from tsl.nn.metrics.metric_base import MaskedMetric

def masked_nseba(y_hat, y, per_basin_stds, reduction, eps=.1):
    num = torch.square(y - y_hat)
    den = torch.square(per_basin_stds + eps)

    nseba = torch.nansum(num / den, dim=0) # sum over
    nseba = torch.squeeze(nseba) # remove window and horizon dimension

    return nseba

class MaskedNSEBA(MaskedMetric):
    def __init__(self,
                 per_basin_stds,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedNSEBA, self).__init__(metric_fn=masked_nseba,
                                          mask_nans=mask_nans,
                                          mask_inf=mask_inf,
                                          compute_on_step=compute_on_step,
                                          dist_sync_on_step=dist_sync_on_step,
                                          process_group=process_group,
                                          dist_sync_fn=dist_sync_fn,
                                          metric_kwargs={'reduction': 'none'},
                                          at=at)
        self.per_basin_stds = torch.from_numpy(per_basin_stds).float()

    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        mask = mask.bool()

        y_hat = torch.where(mask, y_hat, torch.full_like(y_hat, np.nan))
        y = torch.where(mask, y, torch.full_like(y, np.nan))
        per_basin_stds = self.per_basin_stds.expand_as(y)

        res = self.metric_fn(y_hat, y, per_basin_stds)

        return torch.nansum(res), (~torch.isnan(res)).sum()

    def _compute_std(self, y_hat, y):
        _check_same_shape(y_hat, y)

        per_basin_stds = self.per_basin_stds.expand_as(y)

        val = self.metric_fn(y_hat, y, per_basin_stds)
        return val.sum(), val.numel()