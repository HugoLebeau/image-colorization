'''
Save the log of the a*b* distribution of a given dataset in a csv file.
'''

# Allow import from parent directory
import sys, os, inspect
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

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
parser.add_argument('--max-size', type=int, default=None, metavar="MAXSIZE", help="Maximum number of images (default: None).")
parser.add_argument('--seed', type=int, default=1, metavar="SEED", help="Random seed (default: 1).")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

visible_ab = pd.read_csv('cielab/in_visible_gamut.csv', header=None).values # quantized visible a*b* space
q = visible_ab.shape[0] # number of points in the quantized a*b* space
distrib = np.zeros(q, dtype=int)

# Load dataset
transform = lambda img: rgb2ab(transforms.ToTensor()(img))
if args.dataset == "Places205":
    dataset = Places205('datasets/', transform=transform, maxsize=args.max_size)
else:
    raise NameError(args.dataset)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

for batch in tqdm(data_loader):
    for img_ab in batch:
        points = img_ab.permute(1, 2, 0).view(-1, 2) # list all a*b* points in the image
        dmat = cdist(points, visible_ab, metric='euclidean') # compute distances
        idx, counts = np.unique(np.argmin(dmat, axis=1), return_counts=True)
        distrib[idx] += counts # update distribution

np.savetxt('datasets/logdistrib_'+args.dataset+'.csv', np.log(distrib)-np.log(distrib.sum())) # save log-distribution
