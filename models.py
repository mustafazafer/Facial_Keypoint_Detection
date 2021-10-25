## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # image size = (224,224)
        
        self.conv1 = nn.Conv2d(1, 32, 5) #(224-5)/1+1 = 220
        #(32,220,220) after conv1
        #(32,110,110) after maxpool
        
        self.conv2 = nn.Conv2d(32, 64, 5) #(110-5)/1+1 = 106
        #(64,106,106) after conv2
        #(64,53,53) after maxpool
        
        self.conv3 = nn.Conv2d(64, 128, 4) #(53-4)/1+1 = 50
        #(128,50,50) after conv
        #(128,25,25) after maxpool
        
        self.conv4 = nn.Conv2d(128, 64, 4) #(25-4)/1+1 = 22
        #(64,22,22) after conv4
        #(64,11,11) after maxpool
       
        self.pool = nn.MaxPool2d(2,2) 
       
        self.drop_tiny = nn.Dropout(p=0.2)
        self.drop_low = nn.Dropout(p=0.3)
        self.drop_mid = nn.Dropout(p=0.4)
        self.drop_high = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(64*11*11, 1024)
      
        self.fc2 = nn.Linear(1024, 512)
        
        self.fc3 = nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.drop_tiny(x)         
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.drop_low(x)                     
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn4(x)
        x = self.drop_mid(x)
                      
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = self.bn5(x)
        x = self.drop_high(x)                
        x = F.relu(self.fc2(x))
        x = self.bn6(x)
        x = self.drop_high(x)              
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x