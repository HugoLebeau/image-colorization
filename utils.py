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
    img_ab : torch.Tensor, shape (N, 2, H, W)
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
    z : torch.Tensor, shape(N, H, W, Q)
        Soft-encoded images.

    '''
    n, _, h, w = img_ab.shape
    points = img_ab.permute(0, 2, 3, 1).reshape((n*h*w, 2)) # list a*b* points
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(visible_ab)
    distances, indices = nbrs.kneighbors(points) # compute (approximate) nearest neighbors
    kernel = torch.exp(-0.5*torch.tensor(distances)**2/sigma**2)
    soft_encoding = (kernel/kernel.sum(dim=1).reshape((n*h*w, 1))).float() # soft-encoding
    z = torch.zeros((n*h*w, q))
    idx = np.tile(np.arange(n*h*w, dtype=np.int32), (k, 1)).T
    z[idx, indices] = soft_encoding
    z = z.reshape((n, h, w, q))
    return z

def z2ab(z, temp=0.38):
    '''
    Map from class probabilities to point estimates in a*b* space with an
    annealed-mean.

    Parameters
    ----------
    z : torch.Tensor, shape (N, H, W, Q)
        Class probabilities.
    temp : float, optional
        Temperature in ]0, 1]. The default is 0.38.

    Returns
    -------
    torch.Tensor, shape (N, 2, H, W)
        a*b* estimate.

    '''
    n, h, w, _ = z.shape
    temp_z = torch.exp(torch.log(z)/temp)
    ft_z = (temp_z.permute(3, 0, 1, 2)/torch.sum(temp_z, axis=-1)).permute(1, 2, 3, 0)
    prod_a = ft_z*visible_ab[:, 0]
    prod_b = ft_z*visible_ab[:, 1]
    a, b = torch.sum(prod_a, axis=-1).unsqueeze(dim=1), torch.sum(prod_b, axis=-1).unsqueeze(dim=1)
    return torch.cat((a, b), dim=1)

def logdistrib_smoothed(logdistrib, sigma=5.):
    '''
    Smooth a distribution given its log with a gaussian kernel.

    Parameters
    ----------
    logdistrib : torch.Tensor, shape (Q)
        The log of the distribution.
    sigma : float, optional
        Standard deviation of the Gaussian kernel. The default is 5..

    Returns
    -------
    torch.Tensor, shape (Q)
        The log of the smoothed distribution.

    '''
    dmat = torch.tensor(squareform(pdist(visible_ab)))
    logkernel = -0.5*dmat**2/sigma**2
    return torch.logsumexp(logdistrib+logkernel, dim=1)-torch.logsumexp(logkernel, dim=1)

def zero_padding(background_weight, instance_weight, box):
    '''
    Resize instances and do zero-padding to fit them in the background.

    Parameters
    ----------
    background_weight : torch.Tensor, shape (N, C, H, W)
        Background weights.
    instance_weight : list[N] of torch.Tensor, shape (B[i], C, H, W)
        For each image, weights of each instance.
    box : list[N] of torch.Tensor, shape (B[i], 4)
        For each image, the boxes of each instance (x1, y1, x2, y2).

    Returns
    -------
    weight_map : list[N] of torch.Tensor, shape (B[i], C, H, W)
        For each image, the weights of the instances resized to their place in
        the background with zero-padding elsewhere.

    '''
    n, _, h, w = background_weight.shape
    weight_map = list()
    for i in range(n):
        weight_map.append(list())
        for j in range(box[i].shape[0]):
            # Resize every instance weights to their scale in the background
            size = (box[i][j, 3]-box[i][j, 1], box[i][j, 2]-box[i][j, 0])
            resized = torch.nn.functional.interpolate(instance_weight[i][j].unsqueeze(dim=0), size=size, mode='bilinear', align_corners=False)
            # Pad with zeros
            pad = (box[i][j, 0], w-box[i][j, 0]-resized.shape[-1], box[i][j, 3]-resized.shape[-2], h-box[i][j, 3])
            weight_map[i].append(torch.nn.functional.pad(resized, pad, "constant", 0))
        weight_map[i] = torch.cat(weight_map[i])
    return weight_map

def extract(image, box, resize):
    '''
    Extract instances in the given boxes and resize them.

    Parameters
    ----------
    image : torch.Tensor, shape (N, C, H, W)
        A batch of images.
    box : list[N] of torch.Tensor, shape (B[i], 4)
        Boxes of the instances in each image.
    resize : callable
        Function that resizes to (H, W).

    Returns
    -------
    instance : list[N] of torch.Tensor, shape (B[i], C, H, W)
        For each image, a tensor with each instance in shape (H, W).

    '''
    n = len(box)
    instance = list()
    for i in range(n):
        instance.append(list())
        for j in range(box[i].shape[0]):
            extracted = image[i, :, box[i][j, 1]:box[i][j, 3], box[i][j, 0]:box[i][j,2]]
            instance[i].append(resize(extracted).unsqueeze(dim=0))
        instance[i] = torch.cat(instance[i], dim=0)
    return instance
