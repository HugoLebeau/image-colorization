import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

visible_ab = torch.tensor(pd.read_csv('cielab/in_visible_gamut.csv', header=None).values) # quantized visible a*b* space
q = visible_ab.shape[0] # number of points in the quantized visible a*b* space

def ab2z(img_ab, k=5, sigma=5.):
    '''
    In an a*b* image, convert ground truth color Y to vector Z using a soft-
    encoding scheme.

    Parameters
    ----------
    img_ab : torch.tensor, shape (2, H, W)
        a* and b* channels of an image.
    k : int, optional
        Number of nearest neighbors to each ground truth pixel considered in
        the soft-encording. The default is 5.
    sigma : float, optional
        Standard deviation of the gaussian kernel used for the soft-encoding.
        The default is 5..

    Returns
    -------
    z : torch.tensor, shape(H, W, Q)
        Soft-encoded image.

    '''
    _, h, w = img_ab.shape
    points = img_ab.permute(1, 2, 0).view((h*w, 2)) # list a*b* points
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(visible_ab)
    distances, indices = nbrs.kneighbors(points) # compute nearest neighbors
    kernel = torch.exp(-0.5*torch.tensor(distances)**2/sigma**2)
    soft_encoding = kernel/kernel.sum(dim=1).view((h*w, 1)) # soft-encoding
    z = torch.zeros((h*w, q), dtype=torch.float64)
    idx = np.tile(np.arange(h*w, dtype=np.int64), (k, 1)).T
    z[idx, indices] = soft_encoding
    z = z.view((h, w, q))
    return z

def z2ab(z):
    return 0
