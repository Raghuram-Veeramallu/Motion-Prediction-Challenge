import torch
from torch import nn
import numpy as np
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

def nll_with_covariances(gt, predictions, confidences, avails, covariance_matrices):
    precision_matrices = torch.inverse(covariance_matrices)
    gt = torch.unsqueeze(gt, 1)
    avails = avails[:, None, :, None]
    coordinates_delta = (gt - predictions).unsqueeze(-1)
    errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta
    errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * torch.logdet(covariance_matrices).unsqueeze(-1))
    assert torch.isfinite(errors).all()
    with np.errstate(divide="ignore"):
        errors = nn.functional.log_softmax(confidences, dim=1) + \
            torch.sum(errors, dim=[2, 3])
    errors = -torch.logsumexp(errors, dim=-1, keepdim=True)
    return torch.mean(errors)

def pytorch_neg_multi_log_likelihood_batch(gt, predictions, confidences, avails):
    gt = torch.unsqueeze(gt, 1)
    avails = avails[:, None, :, None]
    error = torch.sum(
        ((gt - predictions) * avails) ** 2, dim=-1
    )
    with np.errstate(
        divide="ignore"
    ):
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )
    error = -torch.logsumexp(error, dim=-1, keepdim=True)
    return torch.mean(error)