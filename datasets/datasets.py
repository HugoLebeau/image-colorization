import pandas as pd
from PIL import Image
from torch import utils

class COCOStuff(utils.data.Dataset):
    def __init__(self, path, transform, rgb_only=True, directory='directory_COCO-Stuff.csv', maxsize=None):
        super(COCOStuff, self).__init__()
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

class Places205(utils.data.Dataset):
    def __init__(self, path, transform, rgb_only=True, directory='directory_Places205.csv', maxsize=None):
        super(Places205, self).__init__()
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
