import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self, n_modules, n_convols,  n_targets, in_channels, out_channels, kernel_size, stride, padding):
        super(CNN, self).__init__()

        blocks = []
        for i in range(n_modules):
            in_c = int((i>0)* (out_channels * (2**(i-1))) + (i==0)*in_channels)
            out_c = int(out_channels * (2**i))
            for j in range(n_convols):
                blocks.append(nn.Conv2d((j==0)*in_c + (j>0)*out_c, out_c, kernel_size, stride, padding))
                blocks.append(nn.BatchNorm2d(out_c))
                blocks.append(nn.ReLU())
            blocks.append(nn.Dropout2d(p=0.1))
            blocks.append(nn.MaxPool2d(2))

        self.module = nn.Sequential( *blocks )
        print(self.module)
        self.out = nn.Linear(out_c, n_targets)


    def forward(self, x):
        x = self.module(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(2)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        output = F.log_softmax(output)
        return output


######################################################################3

# cnn complex version
class CNN_complex(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.module_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.Conv2d(32,32,3,1,1),
            nn.Dropout2d(p=0.1),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                
        )
        self.module_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            nn.Conv2d(64,64, 3, 1, 1),
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Dropout2d(p=0.1),
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            nn.MaxPool2d(2),                                
        )
        self.module_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),        
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),        
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Dropout2d(p = 0.1),
            nn.BatchNorm2d(128),        
            nn.ReLU(),
            nn.MaxPool2d(2),                                
        )
        self.module_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),        
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),        
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Dropout2d(p = 0.1),
            nn.BatchNorm2d(256),        
            nn.ReLU(),
            nn.MaxPool2d(2),                                
        )
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.module_1(x)
        x = self.module_2(x)
        x = self.module_3(x)
        x = self.module_4(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(2)
        
        x = x.view(x.size(0),-1)
        output = self.out(x)
        output = F.log_softmax(output)
        return output
    



######################################################################3


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.module_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),        
            nn.ReLU(),
            nn.Conv2d(16,16,3,1,1),
            nn.BatchNorm2d(16),        
            nn.ReLU(),
            nn.Conv2d(16,16,3,1,1),
#            nn.Dropout2d(),
            nn.BatchNorm2d(16),        
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                
        )
        self.module_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.Conv2d(32,32, 3, 1, 1),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
#            nn.Dropout2d(),
            nn.BatchNorm2d(32),        
            nn.ReLU(),
            nn.MaxPool2d(2),                                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.module_1(x)
        x = self.module_2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
#        output = F.log_softmax(output)
        return output
