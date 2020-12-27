import torch

def MCE(prop, target, weights=1.):
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

    Returns
    -------
    torch.Tensor
        The weighted multinomial cross entropy loss.

    '''
    return -torch.sum(weights*torch.sum(target*torch.log(prop), axis=-1), axis=[-1, -2])
