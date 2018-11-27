import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.block_1 = nn.Sequential(nn.BatchNorm2d(4),
                                    nn.Conv2d(4, 8, (3, 3)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(8))

        self.block_2 = nn.Sequential(nn.BatchNorm2d(8),
                                    nn.Conv2d(8, 16, (3, 3)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16),
                                    nn.Dropout(0.4))

        self.block_3 = nn.Sequential(nn.BatchNorm2d(16),
                                    nn.Conv2d(16, 32, (3, 3)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.MaxPool2d((2, 2)),
                                    nn.Dropout(0.4))

        self.par_block_3 = nn.Sequential(nn.Conv2d(16, 32, (3, 3)),
                                         nn.ReLU(),
                                         nn.AvgPool2d((2, 2)))

        self.block_4 = nn.Sequential(nn.BatchNorm2d(32),
                                    nn.Conv2d(32, 64, (3, 3)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d((2, 2)),
                                    nn.Dropout(0.4))

        self.par_block_4 = nn.Sequential(nn.Conv2d(32, 64, (3, 3)),
                                         nn.ReLU(),
                                         nn.AvgPool2d((2, 2)))

        self.block_5 = nn.Sequential(nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 128, (3, 3)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.MaxPool2d((2, 2)),
                                    nn.Dropout(0.4))

        self.par_block_5 = nn.Sequential(nn.Conv2d(64, 128, (3, 3)),
                                         nn.ReLU(),
                                         nn.AvgPool2d((2, 2)))

        self.block_6 = nn.Sequential(nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 64, (3, 3)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d((2, 2)),
                                    nn.Dropout(0.4))

        self.par_block_6 = nn.Sequential(nn.Conv2d(128, 64, (3, 3)),
                                         nn.ReLU(),
                                         nn.AvgPool2d((2, 2)))

        self.block_7 = nn.Sequential(nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 32, (3, 3)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.MaxPool2d((2, 2)),
                                    nn.Dropout(0.4))

        self.par_block_7 = nn.Sequential(nn.Conv2d(64, 32, (3, 3)),
                                         nn.ReLU(),
                                         nn.AvgPool2d((2, 2)))

        self.block_8 = nn.Sequential(nn.BatchNorm2d(32),
                                    nn.Conv2d(32, 16, (3, 3)),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16),
                                    nn.Dropout(0.4),
                                    nn.MaxPool2d((2, 2)))

        self.par_block_8 = nn.Sequential(nn.Conv2d(32, 16, (3, 3)),
                                         nn.ReLU(),
                                         nn.AvgPool2d((2, 2)))

        self.linear = nn.Linear(400, 28)

    def summary(self, input_size=(4, 512, 512)):
        summary(self, input_size)

    def forward(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        x = self.block_1(x)
        x = self.block_2(x)

        output = self.block_3(x)
        par_output = self.par_block_3(x)
        x = output + par_output

        output = self.block_4(x)
        par_output = self.par_block_4(x)
        x = output + par_output

        output = self.block_5(x)
        par_output = self.par_block_5(x)
        x = output + par_output

        output = self.block_6(x)
        par_output = self.par_block_6(x)
        x = output + par_output

        output = self.block_7(x)
        par_output = self.par_block_7(x)
        x = output + par_output

        output = self.block_8(x)
        par_output = self.par_block_8(x)
        x = output + par_output

        x = x.view(x.size()[0], -1)

        x = self.linear(x)

        return x

if __name__ == '__main__':

    obj = Model()
    obj.summary()
    #x = torch.randn(1, 4, 512, 512).float().cuda()
    #print(obj(x).shape)
