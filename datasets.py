import pandas as pd
from PIL import Image
from torch import utils

class Places205(utils.data.Dataset):
    def __init__(self, path, transform=None, rgb_only=True, directory='directory_Places205.csv'):
        super(Places205, self).__init__()
        self.path = path
        self.transform = transform
        dirmode = pd.read_csv(directory, header=None)
        if rgb_only:
            dirmode = dirmode[dirmode[1] == 'RGB']
        self.directory = dirmode[0].values
        self.length = None
    
    def __len__(self):
        if self.length is None:
            self.length = len(self.directory)
        return self.length
    
    def __getitem__(self, index):
        if self.transform is None:
            return Image.open(self.directory[index])
        else:
            return self.transform(Image.open(self.directory[index]))
