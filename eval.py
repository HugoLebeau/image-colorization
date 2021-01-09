import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from lpips import LPIPS
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import normalized_root_mse, structural_similarity, peak_signal_noise_ratio

from transforms import lab2rgb, data_transform
from models import Zhang16, Su20
from utils import z2ab
from datasets.datasets import COCOStuff, Places205, SUN2012

def load_dataset(dataset_name, batch_size, max_size, n_threads, transform=data_transform):
    '''
    Load a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    batch_size : int
        Batch size.
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
    data_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the dataset.

    '''
    if dataset_name == "Places205":
        dataset = Places205('datasets/', transform=transform, maxsize=max_size, return_name=True)
    elif dataset_name == "COCO-Stuff":
        dataset = COCOStuff('datasets/', transform=transform, maxsize=max_size, return_name=True)
    elif dataset_name == "SUN2012":
        dataset = SUN2012('datasets/', transform=transform, maxsize=max_size, return_name=True)
    else:
        raise NameError(dataset_name)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_threads)
    return data_loader

def evaluation(model_name, weights, data_loader, use_cuda=False):
    '''
    Evaluate a model.

    Parameters
    ----------
    model_name : str
        Name of the model to be evaluated.
    weights : str
        Path to weights for the initialisation of the model. If None, weights
        are randomly initialized.
    data_loader : torch.utils.data.dataloader.DataLoader
        Data loader of the dataset.
    use_cuda : bool, optional
        Whether or not to use CUDA. The default is False.

    Raises
    ------
    NameError
        If the model is not recognised.

    Returns
    -------
    df : pandas.DataFrame
        Data frame with the evaluation data.

    '''
    if model_name == "Zhang16":
        model = Zhang16(weights=weights)
        resize = transforms.Resize((256, 256))
        process_output = lambda output, data: lab2rgb(torch.cat((data, resize(z2ab(output))), axis=1))
    elif model_name == "Su20":
        model = Su20(weights=weights)
        process_output = lambda output, data: lab2rgb(torch.cat((data, output), axis=1))
    else:
        raise NameError(model_name)
    if use_cuda:
        print("Using GPU.")
        model.cuda()
    else:
        print("Using CPU.")
    model.eval()
    
    df = pd.DataFrame(columns=['name', 'nrmse', 'ssim', 'psnr', 'lpips'], index=range(len(data_loader.dataset)))
    idx = 0
    lpips_loss = LPIPS(net='alex')
    for ite, (names, (data, target)) in enumerate(tqdm(data_loader)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        images_true = lab2rgb(torch.cat((data, target), axis=1))
        images_test = process_output(output, data)
        lpips_values = lpips_loss(images_true*2.-1., images_test*2.-1.)
        for i, name in enumerate(names):
            image_true = images_true[i].permute(1, 2, 0).numpy()
            image_test = images_test[i].permute(1, 2, 0).numpy()
            nrmse = normalized_root_mse(image_true, image_test)
            ssim = structural_similarity(image_true, image_test, data_range=1., multichannel=True)
            psnr = peak_signal_noise_ratio(image_true, image_test, data_range=1.)
            df.loc[idx, 'name'] = name
            df.loc[idx, 'nrmse'] = nrmse
            df.loc[idx, 'ssim'] = ssim
            df.loc[idx, 'psnr'] = psnr
            df.loc[idx, 'lpips'] = lpips_values[i].item()
            idx += 1
    
    return df

def save(path, args, df):
    '''
    Save the training data in a csv file and the arguments in a txt file.

    Parameters
    ----------
    path : str
        Where to save the files.
    args : argparse.Namespace
        Arguments.
    df : pandas.DataFrame
        Data frame with the evaluation data.

    Returns
    -------
    None.

    '''
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    df.to_csv(path+'eval'+args.model+'_'+now+'.csv')
    pd.Series(vars(args)).to_csv(path+'eval'+args.model+'_'+now+'.txt', header=0)

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
    
    data_loader = load_dataset(args.dataset, args.batch_size, args.max_size, args.n_threads)
    df = evaluation(args.model, args.weights, data_loader, use_cuda)
    save('outputs/', args, df)
