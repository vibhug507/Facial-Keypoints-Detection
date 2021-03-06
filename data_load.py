import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2 as cv


class FacialKeypointsDataset(Dataset):
    
    
    def __init__(self,csv_file,root_dir,transform = None):   
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.key_pts_frame)
    
    def __getitem__(self,index):
        image = mpimg.imread(os.path.join(self.root_dir,self.key_pts_frame.iloc[index,0]))
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
            
        key_pts = self.key_pts_frame.iloc[index, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# Tranforms

class Normalise(object):
    
    # output_size (tuple or int): Desired output size. If tuple, output is
    #        matched to output_size. If int, smaller of image edges is matched
    #        to output_size keeping aspect ratio the same.
    
    def __call__(self,sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        image_copy = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image_copy = image_copy / 255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100.0) / 50.0
        
        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class Normalise2(object):
    
    # output_size (tuple or int): Desired output size. If tuple, output is
    #        matched to output_size. If int, smaller of image edges is matched
    #        to output_size keeping aspect ratio the same.
    
    def __call__(self,sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        image_copy = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        
        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class Normalise3(object):
    
    # output_size (tuple or int): Desired output size. If tuple, output is
    #        matched to output_size. If int, smaller of image edges is matched
    #        to output_size keeping aspect ratio the same.
    
    def __call__(self,sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        image_copy = image_copy / 255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100.0) / 50.0
        
        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self,sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    
    # output_size (tuple or int): Desired output size. If int, square crop
    #       is made.
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__(self,sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}