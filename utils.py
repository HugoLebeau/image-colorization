import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

visible_ab = torch.tensor(pd.read_csv('cielab/in_visible_gamut.csv', header=None).values) # quantized visible a*b* space
q = visible_ab.shape[0] # number of points in the quantized visible a*b* space

def ab2z(img_ab, k=5, sigma=5., algorithm='ball_tree'):
    '''
    Given an a*b* image, convert ground truth color Y to vector Z using a soft-
    encoding scheme.

    Parameters
    ----------
    img_ab : torch.tensor, shape (C, 2, H, W)
        a* and b* channels of a batch of images.
    k : int, optional
        Number of nearest neighbors to each ground truth pixel considered in
        the soft-encording. The default is 5.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for the soft-encoding.
        The default is 5..
    algorithm : str, optional
        Algorithm used to compute the nearest neighbors (ball_tree / kd_tree /
        brute / auto).

    Returns
    -------
    z : torch.tensor, shape(C, H, W, Q)
        Soft-encoded images.

    '''
    c, _, h, w = img_ab.shape
    points = img_ab.permute(0, 2, 3, 1).reshape((c*h*w, 2)) # list a*b* points
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(visible_ab)
    distances, indices = nbrs.kneighbors(points) # compute (approximate) nearest neighbors
    kernel = torch.exp(-0.5*torch.tensor(distances)**2/sigma**2)
    soft_encoding = kernel/kernel.sum(dim=1).reshape((c*h*w, 1)) # soft-encoding
    z = torch.zeros((c*h*w, q), dtype=torch.float64)
    idx = np.tile(np.arange(c*h*w, dtype=np.int64), (k, 1)).T
    z[idx, indices] = soft_encoding
    z = z.reshape((c, h, w, q))
    return z

def z2ab(z, temp=0.38):
    '''
    Map from class probabilities to point estimates in a*b* space with an
    annealed-mean.

    Parameters
    ----------
    z : torch.tensor, shape (H, W, Q)
        Class probabilities.
    temp : float, optional
        Temperature in ]0, 1]. The default is 0.38.

    Returns
    -------
    torch.tensor, shape (H, W)
        a* estimate.
    torch.tensor, shape (H, W)
        b* estimate.

    '''
    h, w, _ = z.shape
    temp_z = torch.exp(torch.log(z)/temp)
    ft_z = (temp_z.permute(2, 0, 1)/torch.sum(temp_z, axis=-1)).permute(1, 2, 0)
    prod_a = ft_z*visible_ab[:, 0]
    prod_b = ft_z*visible_ab[:, 1]
    return (torch.sum(prod_a, axis=-1), torch.sum(prod_b, axis=-1))

def logdistrib_smoothed(logdistrib, sigma=5.):
    '''
    Smooth a distribution given its log with a gaussian kernel.

    Parameters
    ----------
    logdistrib : torch.tensor, shape (Q)
        The log of the distribution.
    sigma : float, optional
        Standard deviation of the Gaussian kernel. The default is 5..

    Returns
    -------
    torch.tensor, shape (Q)
        The log of the smoothed distribution.

    '''
    dmat = torch.tensor(squareform(pdist(visible_ab)))
    logkernel = -0.5*dmat**2/sigma**2
    return torch.logsumexp(logdistrib+logkernel, dim=1)-torch.logsumexp(logkernel, dim=1)
