
import h5py
import numpy as np
import torch
import os
import glob
import torchvision.transforms as tfms
from PIL import Image
import tqdm
import argparse

def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_labels(label_path):

    print(f'Getting labels from {label_path}...')
    mat_data = h5py.File(label_path)
    size = mat_data['/digitStruct/name'].size
    label_max_length = 6
    label_list = -1*np.ones((size,label_max_length))

    for i in tqdm.tqdm(range(size)):
    
        label = get_box_data(i, mat_data)['label']
        if len(label) > label_max_length:
            label_max_length = len(label)
        
        for j, l in enumerate(label):
            if l==10:
                l=0 #HACK: not sure why it puts 10 for 0...
            label_list[i,j] = l

    return label_list.astype(int)

def get_images(image_dir):

    print(f'Getting images from {image_dir}...')
    images = glob.glob(os.path.join(image_dir,'*.png'))
    images = sorted(images,key=lambda x: int(x.split('/')[-1].split('.')[0]))

    transform = tfms.Compose([
            tfms.ToTensor(),
            tfms.Resize((96,128),antialias=True)
        ])

    tensors = []
    for image_path in tqdm.tqdm(images):

        img = Image.open(image_path)
        img = transform(img)
        img = img/255.0 ## normalize

        tensors.append(img)

    return torch.stack(tensors).numpy()

def preprocess_svhn():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-i', type=str, help='Path to the image directory', required=True)
    parser.add_argument('--label_path', '-l', type=str, help='Path to the label file', required=False, default=None)
    parser.add_argument('--out_path', '-o', type=str, help='Path to the output h5 file', required=True)
    args = parser.parse_args()

    if args.label_path is None:
        args.label_path = os.path.join(args.image_dir,'digitStruct.mat')

    labels = get_labels(args.label_path)
    images = get_images(args.image_dir)

    assert len(labels)==len(images), 'Number of labels and images do not match!'

    print(f'Writing to {args.out_path}...')
    ### Write to h5
    with h5py.File(args.out_path,'w') as o:
        o.create_dataset('images',data=images)
        o.create_dataset('labels',data=labels)

if __name__=='__main__':
    preprocess_svhn()