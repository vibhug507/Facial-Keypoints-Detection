import os
import numpy as np
import pandas as pd
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalise, ToTensor, Normalise2, Normalise3
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

key_pts_train = pd.read_csv('data/training_frames_keypoints.csv')
key_pts_test = pd.read_csv('data/test_frames_keypoints.csv')

key_pts_train.columns = [i for i in range(137)]
key_pts_test.columns = [i for i in range(137)]

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalise(),
                                     ToTensor()])

data_transform2 = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalise2(),
                                     ToTensor()])

data_transform3 = transforms.Compose([Normalise3()])

train_dataset = FacialKeypointsDataset(csv_file = 'data/training_frames_keypoints.csv',
                                             root_dir = 'data/training/',
                                             transform = data_transform2)

test_dataset = FacialKeypointsDataset(csv_file = 'data/test_frames_keypoints.csv',
                                      root_dir = 'data/test/',
                                     transform = data_transform)

batch_size = 32

train_loader = DataLoader(train_dataset, 
                          batch_size = batch_size,
                          shuffle = False)

test_loader = DataLoader(test_dataset, 
                          batch_size = batch_size,
                          shuffle = False)

seq = iaa.Sequential([iaa.Fliplr(0.5),
                          iaa.Flipud(0.5),
                          iaa.GaussianBlur(sigma=(0, 3.0))])

def augment(image, keypoints):
                      
    kps = KeypointsOnImage([Keypoint(x = keypoints[i][0],y = keypoints[i][1]) for i in range(68)],image.shape)
    imageAug, kpsAug = seq(image = image, keypoints = kps)
                      
    return imageAug, kpsAug

originalImages = []
augImages = []
originalKeypoints = []
augKeypoints = []


for i, sample in enumerate(train_loader):
        
        images = sample['image']
        key_pts = sample['keypoints']

        images = images.type(torch.FloatTensor)
        images = images.numpy()
        key_pts = key_pts.numpy()
        
        # print(images.shape)  (32,1,224,224)
        # print(key_pts.shape) (32,68,2)
        
        for j in range(len(images)):
            originalImages.append(images[j])
            originalKeypoints.append(key_pts[j])
            
            img = images[j].transpose((1,2,0))
            imageAug, kpsAug = augment(img,key_pts[j])
            imageAug = imageAug.transpose((2,0,1))
            augImages.append(imageAug)
            augKeypoints.append(kpsAug)
            

originalImages = np.array(originalImages)
originalKeypoints = np.array(originalKeypoints)

newKeypoints = [[] for i in range(len(augImages))]

for i in range(len(augKeypoints)):
    for j in range(68):
        x,y = augKeypoints[i][j].x, augKeypoints[i][j].y
        newKeypoints[i].append([x,y])

augImages = np.array(augImages)
augKeypoints = np.array(newKeypoints)


os.chdir("/Users/vibhugarg/Desktop/Facial Keypoints Detection/Original")
for i in range(len(key_pts_train)):
    cv.imwrite(key_pts_train.loc[i,0].strip(), originalImages[i][0])
    
os.chdir("/Users/vibhugarg/Desktop/Facial Keypoints Detection/Augmented")
for i in range(len(key_pts_train)):
    cv.imwrite((key_pts_train.loc[i,0] + "2.jpg").strip(), augImages[i][0])
    
os.chdir("/Users/vibhugarg/Desktop/Facial Keypoints Detection")

trainImages = []


for i in range(len(originalImages)):
    l = [key_pts_train.loc[i,0].strip()]
    x = originalKeypoints.reshape(1,-1)
    for j in range(136):
        l.append(x[0][j])
    trainImages.append(l)
    
for i in range(len(originalImages)):
    l = [(key_pts_train.loc[i,0] + "2,jpg").strip()]
    x = augKeypoints.reshape(1,-1)
    for j in range(136):
        l.append(x[0][j])
    trainImages.append(l)

trainImages = np.array(trainImages)
trainFrame = pd.DataFrame(trainImages, columns = [i for i in range(137)])

trainFrame.to_csv('newData.csv', index = False)