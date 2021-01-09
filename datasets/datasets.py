import pandas as pd
from PIL import Image
from torch import utils

class Dataset(utils.data.Dataset):
    def __init__(self, path, transform, directory, rgb_only, maxsize):
        super(Dataset, self).__init__()
        self.path = path
        self.transform = transform
        dirmode = pd.read_csv(self.path+directory, header=None)
        if rgb_only:
            dirmode = dirmode[dirmode[1] == 'RGB']
        if maxsize:
            dirmode = dirmode[:maxsize]
        self.directory = dirmode[0].values
        self.length = None
    
    def __len__(self):
        if self.length is None:
            self.length = len(self.directory)
        return self.length
    
    def __getitem__(self, index):
        return self.transform(Image.open(self.path+self.directory[index]))

class COCOStuff(Dataset):
    def __init__(self, path, transform, directory='directory_COCO-Stuff.csv', rgb_only=True, maxsize=None):
        super(COCOStuff, self).__init__(path, transform, directory, rgb_only, maxsize)

class Places205(Dataset):
    def __init__(self, path, transform, directory='directory_Places205.csv', rgb_only=True, maxsize=None):
        super(Places205, self).__init__(path, transform, directory, rgb_only, maxsize)

class SUN2012(Dataset):
    def __init__(self, path, transform, directory='directory_SUN2012.csv', rgb_only=True, maxsize=None):
        super(SUN2012, self).__init__(path, transform, directory, rgb_only, maxsize)
