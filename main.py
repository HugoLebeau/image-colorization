import argparse
import torch
import numpy as np

from datasets import Places205
from models import Zhang16
from transforms import data_transform

parser = argparse.ArgumentParser(description="Automatic image colorization")
parser.add_argument('--batch-size', type=int, default=4, metavar="BATCHSIZE",
                    help="Training batch size (default: 4).")
parser.add_argument('--val-part', type=float, default=0.1, metavar='VALPART',
                    help="Portion of the dataset kept for validation (default: 0.1).")
parser.add_argument('--n-threads', type=int, default=0, metavar="NTHREADS",
                    help="How many subprocesses to use for data loading (default: 0).")
parser.add_argument('--seed', type=int, default=1, metavar="SEED",
                    help="Random seed (default: 1).")
args = parser.parse_args()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    dataset = Places205('Places205/', transform=data_transform)
    dataset_size = len(dataset)
    val_size = int(args.val_part*dataset_size)
    train_size = dataset_size-val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.n_threads)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.n_threads)
    
    model = Zhang16()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")
    