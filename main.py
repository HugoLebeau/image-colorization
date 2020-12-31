import argparse
import torch
import numpy as np
from datetime import datetime
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from loss_functions import MCE
from models import model_init, Zhang16
from transforms import data_transform
from utils import ab2z, logdistrib_smoothed
from datasets.datasets import Places205

def load_dataset(dataset_name, val_part, batch_size, max_size, n_threads, transform=data_transform):
    '''
    Load a dataset and perform a train/val split.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    val_part : float
        Portion of the dataset kept for validation.
    batch_size : int
        Training batch size.
    max_size : int
        Maximum number of images. If None, the full dataset is loaded.
    n_threads : int
        How many subprocesses to use for data loading.
    transform : callable, optional
        Transformation applied to the data. The default is None.

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
    else:
        raise NameError(dataset_name)
    dataset_size = len(dataset)
    val_size = int(val_part*dataset_size)
    train_size = dataset_size-val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_threads)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_threads)
    return train_loader, val_loader, proba_ab

def training(model_name, train_loader, val_loader, proba_ab, use_cuda=False):
    '''
    Load a model and train it with the given data.

    Parameters
    ----------
    model_name : str
        Name of the model to be trained.
    train_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the training set.
    val_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the validation set.
    proba_ab : torch.tensor
        Distribution of a*b* visible values in the dataset.
    use_cuda : bool, optional
        Whether or not to use CUDA. The default is False.

    Raises
    ------
    NameError
        If the model is not recognised.

    Returns
    -------
    model : torch.nn.Module
        Trained model.
    training_loss : torch.Tensor
        Training loss at each iteration.
    validation_loss : torch.Tensor
        Validation loss.

    '''
    if model_name == "Zhang16":
        model = Zhang16()
        if use_cuda:
            print("Using GPU.")
            model.cuda()
        else:
            print("Using CPU.")
        
        optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.99), weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
        w = 1./(0.5*proba_ab+0.5/proba_ab.shape[0])
        w /= (proba_ab*w).sum()
        resize = transforms.Resize((64, 64))
        def criterion(prop, target):
            z_target = ab2z(resize(target.cpu()), k=5, sigma=5.)
            return MCE(prop.cpu(), z_target, weights=w[z_target.argmax(dim=-1)]).sum()
    else:
        raise NameError(model_name)
    model_init(model)
    
    training_loss, validation_loss = torch.zeros(len(train_loader)), 0
    ok = True
    # TRAINING
    model.train()
    for it, (data, target) in enumerate(tqdm(train_loader)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if np.isnan(loss.data.item()):
            ok = False
            break
        loss.backward()
        training_loss[it] = loss.data.item()/data.shape[0]
        optimizer.step()
        scheduler.step(training_loss[it])
    # VALIDATION
    model.eval()
    if ok:
        for data, target in tqdm(val_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            validation_loss += criterion(output, target).data.item()
        val_size = len(val_loader)
        if val_size > 0:
            validation_loss /= val_size
    
    return model, training_loss, validation_loss

def save(path, model_name, model, training_loss, validation_loss):
    '''
    Save the model weights in a pth file and the training/validations losses
    in a csv file.

    Parameters
    ----------
    model_name : str
        Name of the model.
    model : torch.nn.Module
        Model whose weights are to be saved.
    training_loss : torch.Tensor
        Training loss at each iteration.
    validation_loss : torch.Tensor
        Validation loss.

    Returns
    -------
    None.

    '''
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(model.state_dict(), path+model_name+'_'+now+'.pth')
    file = open(path+model_name+'_'+now+'.csv', 'w')
    file.write("Validation loss,{}\n".format(validation_loss))
    file.write("Iteration,Training loss\n")
    n_it = len(training_loss)
    for it in range(n_it):
        file.write("{},{}\n".format(it+1, training_loss[it]))
    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automatic image colorization")
    parser.add_argument('--dataset', type=str, metavar="DATASET",
                        help="Dataset used for training.")
    parser.add_argument('--model', type=str, metavar="MODEL",
                        help="Model to be trained.")
    parser.add_argument('--batch-size', type=int, default=4, metavar="BATCHSIZE",
                        help="Training batch size (default: 4).")
    parser.add_argument('--val-part', type=float, default=0.1, metavar='VALPART',
                        help="Portion of the dataset kept for validation (default: 0.1).")
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
    
    train_loader, val_loader, proba_ab = load_dataset(args.dataset, args.val_part, args.batch_size, args.max_size, args.n_threads)
    model, training_loss, validation_loss = training(args.model, train_loader, val_loader, proba_ab, use_cuda)
    save('outputs/', args.model, model, training_loss, validation_loss)
