import numpy as np
import torch
import h5py
import os
from torch.utils.data import Dataset
import torch.nn.functional as func
import torchvision.models as models
import torchvision.transforms as tfms
from PIL import Image
from skimage.draw import random_shapes
import glob


class StreetNumberDataset(Dataset):
    def __init__(self, input_path, Ndigits=2):

        with h5py.File(input_path, 'r') as f:
            self.images = f['images'][:]
            self.labels = f['labels'][:]
    
        '''
        #self.images = glob.glob(image_dir + "/*.png")
        self.labels = np.load(label_file)
        '''
        
        ### select subset of data which have exactly Ndigits in the label
        Ndigits_array = np.count_nonzero((self.labels>=0),axis=-1)
        mask = np.where(Ndigits_array == Ndigits)[0]

        self.labels = self.labels[mask][:,:Ndigits]
        self.images = self.images[mask]

    def __len__(self):
       
        return len(self.images)

    ### converts image to suitable format for pytorch
    def transform_image(self, image):
        transform = tfms.Compose([
            tfms.ToTensor(),
            tfms.Resize((96,128),antialias=True)
        ])
        return transform(image)

    def __getitem__(self, idx):

        # img = Image.open(self.images[idx])
        # img = self.transform_image(img)
        #img_idx = int(self.images[idx].split('/')[-1].split('.')[0]) - 1
        # img = img/255.0 ## normalize

        img = torch.FloatTensor(self.images[idx])
        lab = torch.LongTensor(self.labels[idx])
        #lab = lab[0] ## HACK: simplification (only classify first digit)
        
        return img, lab