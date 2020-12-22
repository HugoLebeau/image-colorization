import torch
import pandas as pd

visible_ab = torch.tensor(pd.read_csv('cielab/in_visible_gamut.csv', header=None).values)
q = visible_ab.shape[0]

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
    points = img_ab.permute(1, 2, 0)
    dist2 = torch.sum((points.expand(q, -1, -1, -1).permute(1, 2, 0, 3)-visible_ab.expand(h, w, -1, -1))**2, axis=-1)
    mins = torch.topk(dist2, k=k, dim=-1, largest=False)
    soft_encoding = (torch.exp(-0.5*mins.values/sigma**2).permute(2, 0, 1)/torch.exp(-0.5*mins.values/sigma**2).sum(axis=-1)).permute(1, 2, 0)
    z = torch.zeros((h, w, q))
    for hi in range(h):
        for wi in range(w):
            idx = mins.indices[hi, wi]
            z[hi, wi, idx] = soft_encoding[hi, wi]
    return z

def z2ab(z):
    return 0
