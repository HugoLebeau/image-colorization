''' List the modes of each image (RGB, L, ...) of a given dataset in a csv file. '''

import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Automatic image colorization")
parser.add_argument('--path', type=str, metavar="PATH", help="Path to the dataset.")
args = parser.parse_args()

list_images = [root+'/'+file for root, dirs, files in os.walk(args.path) for file in files]
modes = pd.Series(index=list_images, dtype=object)

for img in tqdm(modes.index):
    modes[img] = Image.open(img).mode

modes.to_csv('image_modes.csv', header=False)
