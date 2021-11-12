## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 128,kernel_size = 3)
        self.batch1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv2 = nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 3)
        self.batch2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv3 = nn.Conv2d(in_channels = 256,out_channels = 512,kernel_size = 3)
        self.batch3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        
        #self.drop1 = nn.Dropout(p = 0.4)
        
        self.conv4 = nn.Conv2d(in_channels = 512,out_channels = 1024,kernel_size = 3)
        self.batch4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv5 = nn.Conv2d(in_channels = 1024,out_channels = 2048,kernel_size = 3)
        self.batch5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv6 = nn.Conv2d(in_channels = 2048,out_channels = 4096,kernel_size = 3)
        self.batch5 = nn.BatchNorm2d(512)
        self.act6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size = 2)
        
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features = 4096, out_features =  1024)
        self.drop2 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 1024)
        self.fc3 = nn.Linear(in_features = 1024, out_features = 136)
        
    def forward(self,x):
        out = self.conv1(x)
        #out = self.batch1(out)
        out = self.act1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        #out = self.batch2(out)
        out = self.act2(out)
        out = self.pool2(out)
        
        out = self.conv3(out)
        #out = self.batch3(out)
        out = self.act3(out)
        out = self.pool3(out)
        
        #out = self.drop1(out)
        
        out = self.conv4(out)
        #out = self.batch4(out)
        out = self.act4(out)
        out = self.pool4(out)
        
        out = self.conv5(out)
        #out = self.batch5(out)
        out = self.act5(out)
        out = self.pool5(out)
        
        out = self.conv6(out)
        #out = self.batch5(out)
        out = self.act6(out)
        out = self.pool6(out)
        
        out = self.flat(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.fc2(out)
        #out = self.drop3(out)
        out = self.fc3(out)
        
        return out