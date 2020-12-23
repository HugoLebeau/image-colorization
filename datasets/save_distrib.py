'''
Save the a*b* distribution of a given dataset in a csv file.
'''

import argparse
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.distance import cdist

from datasets.datasets import Places205
from transforms import rgb2ab

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, metavar="DATASET", help="Name of the dataset.")
parser.add_argument('--batch-size', type=int, default=4, metavar="BATCHSIZE", help="Batch size (default: 4).")
args = parser.parse_args()

visible_ab = pd.read_csv('cielab/in_visible_gamut.csv', header=None).values # quantized visible a*b* space
q = visible_ab.shape[0] # number of points in the quantized a*b* space
distrib = np.zeros(q, dtype=int)

# Load dataset
transform = lambda img: rgb2ab(transforms.ToTensor()(img))
if args.dataset == "Places205":
    dataset = Places205('./', transform=transform)
else:
    raise NameError(args.dataset)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

for batch in tqdm(data_loader):
    for img_ab in batch:
        points = img_ab.permute(1, 2, 0).view(-1, 2) # list all a*b* points in the image
        dmat = cdist(points, visible_ab, metric='euclidean') # compute distances
        distrib[np.argmin(dmat, axis=1)] += 1 # update distribution

np.savetxt('distrib_'+args.dataset+'.csv', distrib, fmt='%d') # save distribution
