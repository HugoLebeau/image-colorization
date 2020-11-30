import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from transforms import rgb2lab, lab2rgb, data_transform

parser = argparse.ArgumentParser(description="Automatic image colorization")
parser.add_argument('--batch-size', type=int, default=4, metavar="BATCHSIZE",
                    help="Training batch size (default: 4).")
parser.add_argument('--n-threads', type=int, default=0, metavar="NTHREADS",
                    help="How many subprocesses to use for data loading (default: 0).")
parser.add_argument('--seed', type=int, default=1, metavar="SEED",
                    help="Random seed (default: 1).")
args = parser.parse_args()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    cifar10_train = torchvision.datasets.CIFAR10('CIFAR10/', train=True, download=True,
                                                 transform=data_transform['color'])
    data_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=4, shuffle=True,
                                              num_workers=args.n_threads)
    
    batch = iter(data_loader).next()
    img = batch[0][1]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
