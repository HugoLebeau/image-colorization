import torch

def MCE(prop, target, weights=1., eps=1e-42):
    '''
    Weighted multinomial cross entropy loss.

    Parameters
    ----------
    prop : torch.Tensor, shape (..., H, W, Q)
        Proposal distribution.
    target : torch.Tensor, shape (..., H, W, Q)
        Target distribution.
    weights : float or torch.Tensor, shape (..., H, W)
        Weights.
    eps : float, optional
        Small value added before computing the log to avoid NaNs. The default
        is 1e-42.

    Returns
    -------
    torch.Tensor
        The weighted multinomial cross entropy loss.

    '''
    return -torch.sum(weights*torch.sum(target*torch.log(prop+eps), axis=-1), dim=(-1, -2))

def smoothL1(prop, target, delta=1.):
    '''
    Smooth-L1 (Huber) loss.

    Parameters
    ----------
    prop : torch.Tensor, shape (..., C, H, W)
        Proposal image.
    target : torch.Tensor, shape (..., C, H, W)
        Target image.
    delta : float, optional
        Steepness parameter. The default is 1..

    Returns
    -------
    torch.Tensor
        The smooth-L1 loss.

    '''
    diff = torch.abs(prop-target)
    pointwise_loss = torch.where(diff < delta, 0.5*torch.pow(diff, 2.), delta*(diff-0.5*delta))
    return torch.sum(pointwise_loss, dim=(-1, -2, -3))
