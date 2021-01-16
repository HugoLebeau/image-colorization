import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from loss_functions import MCE, smoothL1
from models import Zhang16, Su20, Su20Zhang16Instance
from transforms import data_transform
from utils import ab2z, z2ab, logdistrib_smoothed, extract
from datasets.datasets import COCOStuff, Places205, SUN2012

def load_dataset(dataset_name, val_size, batch_size, max_size, n_threads, transform=data_transform):
    '''
    Load a dataset and perform a train/val split.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    val_size : int
        Size of the validation set.
    batch_size : int
        Training batch size.
    max_size : int
        Maximum number of images. If None, the full dataset is loaded.
    n_threads : int
        How many subprocesses to use for data loading.
    transform : callable, optional
        Transformation applied to the data. The default is data_transform.

    Raises
    ------
    NameError
        If the dataset is not recognised.

    Returns
    -------
    train_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the training set.
    val_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the validation set.
    proba_ab : torch.tensor
        Distribution of a*b* visible values in the dataset.

    '''
    if dataset_name == "Places205":
        dataset = Places205('datasets/', transform=transform, maxsize=max_size)
        proba_ab = torch.exp(logdistrib_smoothed(torch.tensor(np.loadtxt('datasets/logdistrib_Places205.csv'))))
    elif dataset_name == "COCO-Stuff":
        dataset = COCOStuff('datasets/', transform=transform, maxsize=max_size)
        proba_ab = torch.exp(logdistrib_smoothed(torch.tensor(np.loadtxt('datasets/logdistrib_Places205.csv'))))
    elif dataset_name == "SUN2012":
        dataset = SUN2012('datasets/', transform=transform, maxsize=max_size)
        proba_ab = torch.exp(logdistrib_smoothed(torch.tensor(np.loadtxt('datasets/logdistrib_Places205.csv'))))
    else:
        raise NameError(dataset_name)
    dataset_size = len(dataset)
    train_size = dataset_size-val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_threads)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_threads)
    return train_loader, val_loader, proba_ab

def training(model_name, weights, lr, train_loader, val_loader, val_size, val_step, proba_ab, use_cuda=False, max_nan=20):
    '''
    Load a model and train it with the given data.

    Parameters
    ----------
    model_name : str
        Name of the model to be trained.
    weights : str
        Path to weights for the initialisation of the model. If None, weights
        are randomly initialized.
    lr : float
        Learning rate.
    train_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the training set.
    val_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the validation set.
    val_size : int
        Size of the validation set.
    val_step : int
        Number of training iterations before a validation is performed.
    proba_ab : torch.tensor
        Distribution of a*b* visible values in the dataset.
    use_cuda : bool, optional
        Whether or not to use CUDA. The default is False.
    max_nan : int, optional
        Maximum number of consecutive NaN gradients before break. The default
        is 20.

    Raises
    ------
    NameError
        If the model is not recognised.

    Returns
    -------
    model : torch.nn.Module
        Trained model.
    df : pandas.DataFrame
        Data frame with the training data.

    '''
    if model_name == "Zhang16":
        model = Zhang16(weights=weights)
        if use_cuda:
            print("Using GPU.")
            model.cuda()
        else:
            print("Using CPU.")
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-3)
        w = 1./(0.5*proba_ab+0.5/proba_ab.shape[0])
        w /= (proba_ab*w).sum()
        resize = transforms.Resize((64, 64))
        def criterion(prop, target):
            z_target = ab2z(resize(target.cpu()))
            return MCE(prop.cpu(), z_target, weights=w[z_target.argmax(dim=-1)]).sum()
    elif model_name == "Su20Instance":
        model = Su20Zhang16Instance(weights=weights)
        if use_cuda:
            print("Using GPU.")
            model.cuda()
        else:
            print("Using CPU.")
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999), weight_decay=1e-3)
        w = 1./(0.5*proba_ab+0.5/proba_ab.shape[0])
        w /= (proba_ab*w).sum()
        resize = transforms.Resize((64, 64))
        def criterion(output, target):
            prop, box = output
            target_instance = extract(target.cpu(), box, resize)
            loss, ok = 0., False
            for i, img in enumerate(target_instance):
                if prop[i] is not None:
                    z_target = ab2z(img)
                    loss += MCE(prop[i].cpu(), z_target, weights=w[z_target.argmax(dim=-1)]).mean()
                    ok = True
            if not ok:
                loss = torch.tensor(0., requires_grad=True)
            return loss
    elif model_name == "Su20":
        model = Su20(weights=weights)
        if use_cuda:
            print("Using GPU.")
            model.cuda()
        else:
            print("Using CPU.")
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999), weight_decay=1e-3)
        resize = transforms.Resize((64, 64))
        def criterion(output, target):
            prop = z2ab(output.cpu())
            return smoothL1(prop, resize(target.cpu())).sum()
    else:
        raise NameError(model_name)
    
    n_ite = len(train_loader)
    df = pd.DataFrame(columns=['lr', 'training loss', 'validation loss', 'optimizer step'], index=range(n_ite))
    before_val = val_step
    count_nan = 0
    # TRAINING
    model.train()
    for ite, (data, target) in enumerate(tqdm(train_loader)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        df['lr'][ite] = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        df['training loss'][ite] = loss.data.item()/data.shape[0]
        count_nan += 1
        for param in model.parameters():
            param.grad[torch.isnan(param.grad)] == 0.
        if not np.any([torch.any(torch.isnan(param.grad)).item() for param in model.parameters() if param.grad is not None]): # if the gradients are not nan
            count_nan = 0
            df['optimizer step'][ite] = True
            optimizer.step()
        if count_nan >= max_nan:
            break
        before_val -= data.shape[0]
        if before_val <= 0 or ite == n_ite-1: # VALIDATION
            before_val = val_step
            df['validation loss'][ite] = 0.
            model.eval()
            for data, target in val_loader:
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                df['validation loss'][ite] += criterion(output, target).data.item()
            df['validation loss'][ite] /= val_size
            model.train()
    
    return model, df

def save(path, args, model, df):
    '''
    Save the model weights in a pth file, the training data in a csv file and
    the arguments in a txt file.

    Parameters
    ----------
    path : str
        Where to save the files.
    args : argparse.Namespace
        Arguments that produced the model.
    model : torch.nn.Module
        Model whose weights are to be saved.
    df : pandas.DataFrame
        Data frame with the training data.

    Returns
    -------
    None.

    '''
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(model.state_dict(), path+args.model+'_'+now+'.pth')
    df.to_csv(path+args.model+'_'+now+'.csv')
    pd.Series(vars(args)).to_csv(path+args.model+'_'+now+'.txt', header=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automatic image colorization")
    parser.add_argument('--dataset', type=str, metavar="DATASET",
                        help="Dataset used for training.")
    parser.add_argument('--model', type=str, metavar="MODEL",
                        help="Model to be trained.")
    parser.add_argument('--weights', type=str, default=None, metavar="WEIGHTS",
                        help="Path to weights for the initialisation of the model (default: None).")
    parser.add_argument('--batch-size', type=int, default=32, metavar="BATCHSIZE",
                        help="Training batch size (default: 32).")
    parser.add_argument('--lr', type=float, default=1e-5, metavar="LR",
                        help="Learning rate (default: 1e-5).")
    parser.add_argument('--val-size', type=int, default=10000, metavar='VALSIZE',
                        help="Size of the validation set (default: 10 000).")
    parser.add_argument('--val-step', type=float, default=100000, metavar='VALSIZE',
                        help="Number of training iterations before a validation is performed (default: 100 000).")
    parser.add_argument('--max-size', type=int, default=None, metavar="MAXSIZE",
                        help="Maximum number of images (default: None).")
    parser.add_argument('--n-threads', type=int, default=0, metavar="NTHREADS",
                        help="How many subprocesses to use for data loading (default: 0).")
    parser.add_argument('--seed', type=int, default=1, metavar="SEED",
                        help="Random seed (default: 1).")
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_loader, val_loader, proba_ab = load_dataset(args.dataset, args.val_size, args.batch_size, args.max_size, args.n_threads)
    model, df = training(args.model, args.weights, args.lr, train_loader, val_loader, args.val_size, args.val_step, proba_ab, use_cuda)
    save('outputs/', args, model, df)
