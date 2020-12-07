import argparse
import torch
import numpy as np
from datetime import datetime
from torch import optim
from tqdm import tqdm

from datasets import Places205
from models import Zhang16
from transforms import data_transform

parser = argparse.ArgumentParser(description="Automatic image colorization")
parser.add_argument('--dataset', type=str, metavar="DATASET",
                    help="Dataset used for training.")
parser.add_argument('--model', type=str, metavar="MODEL",
                    help="Model to be trained.")
parser.add_argument('--batch-size', type=int, default=4, metavar="BATCHSIZE",
                    help="Training batch size (default: 4).")
parser.add_argument('--val-part', type=float, default=0.1, metavar='VALPART',
                    help="Portion of the dataset kept for validation (default: 0.1).")
parser.add_argument('--n-threads', type=int, default=0, metavar="NTHREADS",
                    help="How many subprocesses to use for data loading (default: 0).")
parser.add_argument('--seed', type=int, default=1, metavar="SEED",
                    help="Random seed (default: 1).")
args = parser.parse_args()

def load_dataset(dataset_name):
    '''
    Load a dataset and perform a train/val split.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Raises
    ------
    NameError
        If the dataset is not recognised.

    Returns
    -------
    train_size : int
        Size of the training set.
    val_size : int
        Size of the validation set.
    train_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the training set.
    val_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the validation set.

    '''
    if dataset_name == "Places205":
        dataset = Places205('Places205/', transform=data_transform)
    else:
        raise NameError(dataset_name)
    dataset_size = len(dataset)
    val_size = int(args.val_part*dataset_size)
    train_size = dataset_size-val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.n_threads)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.n_threads)
    return train_size, val_size, train_loader, val_loader

def training(model_name, train_loader, val_loader, use_cuda=False):
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
    training_loss : ndarray, shape (n_epochs,)
        Training loss at each epoch.
    validation_loss : ndarray, shape (n_epochs,)
        Validation loss at each epoch.

    '''
    if model_name == "Zhang16":
        model = Zhang16()
        if use_cuda:
            print("Using GPU")
            model.cuda()
        else:
            print("Using CPU")
        
        lr_steps = [(3e-5, 200000), (1e-5, 375000), (3e-6, 450000)]
        n_epochs = lr_steps[-1][1]
        def lr_lambda(epoch):
            n, lr = len(lr_steps), None
            for i in range(n):
                if epoch < lr_steps[i][1]:
                    lr = lr_steps[i][0]
                    break
            return lr
        optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.99), weight_decay=1e-3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = ?
    else:
        raise NameError(model_name)
    
    training_loss, validation_loss = np.zeros(n_epochs), np.zeros(n_epochs)
    epochs = np.arange(n_epochs)
    for epoch in tqdm(epochs):
        # TRAINING
        model.train()
        training_loss.append
        for data, target in train_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            training_loss[epoch] += loss.data.item()
            optimizer.step()
        scheduler.step()
        training_loss[epoch] /= train_size
        # VALIDATION
        model.eval()
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            validation_loss[epoch] += criterion(output, target).data.item()
        if val_size > 0:
            validation_loss[epoch] /= val_size
    
    return model, training_loss, validation_loss

def save(model, training_loss, validation_loss):
    '''
    Save the model weights in a pth file and the training/validations losses
    in a csv file.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose weights are to be saved.
    training_loss : ndarray, shape (n_epochs,)
        Training loss at each epoch.
    validation_loss : ndarray, shape (n_epochs,)
        Validation loss at each epoch.

    Returns
    -------
    None.

    '''
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(model.state_dict(), args.model+'_'+now+'.pth')
    file = open(args.model+'_'+now+'.csv', 'w')
    file.write("Epoch,Training loss,Validation loss\n")
    n_epochs = len(training_loss)
    for i in range(n_epochs):
        file.write("{},{},{}\n".format(i+1, training_loss[i], validation_loss[i]))
    file.close()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_size, val_size, train_loader, val_loader = load_dataset(args.dataset)
    model, training_loss, validation_loss = training(args.model, train_loader, val_loader, use_cuda)
    save(model, training_loss, validation_loss)
