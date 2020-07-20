"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from exercise_code.data.facial_keypoints_dataset import FacialKeypointsDataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=7),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
        )
        self.dp=nn.Dropout()
        self.fc1 = nn.Linear(20*20*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,30)
        
        
        
            
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dp(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x
    
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)


        

    
    def training_step(self, batch, batch_idx):
        
        inp = batch['image']
        labels = batch['keypoints']
        
        y = self.forward(inp)
        y = y.reshape(512,15,2)
        loss=nn.MSELoss()
        loss=loss(labels,y)
        #loss = self.cross_entropy_loss(y,labels)
        logs = { 'train_loss': loss }
        return {'loss': loss, 'log': logs}
    
 
    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])
        return optim
    
    def prepare_dataset(self):
        
        i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
        data_root = os.path.join(i2dl_exercises_path, "datasets", "facial_keypoints")
        
        train_dataset = FacialKeypointsDataset(
        train=True,
        transform=transforms.ToTensor(),
        root=data_root,
        download_url=download_url,
        )
        
      
    
    
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=512)
    
class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
