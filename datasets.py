import os
from PIL import Image
from torch import utils

class Places205(utils.data.Dataset):
    def __init__(self, path, transform=None):
        super(Places205, self).__init__()
        self.path = path
        self.transform = transform
        self.list_files = [root+'/'+file for root, dirs, files in os.walk(path) for file in files]
        self.length = None
    
    def __len__(self):
        if self.length is None:
            self.length = len(self.list_files)
        return self.length
    
    def __getitem__(self, index):
        if self.transform is None:
            return Image.open(self.list_files[index])
        else:
            return self.transform(Image.open(self.list_files[index]))
