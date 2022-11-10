import MinkowskiEngine as ME

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from SingleParticleDataAccess import LABELS, load_batch

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tqdm

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            # ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=32,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU())
        self.mp1 = nn.Sequential(
            ME.MinkowskiMaxPooling(
                kernel_size = 2,
                stride = 1,
                dimension = D),
            )
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())

        self.mp2 = nn.Sequential(
            ME.MinkowskiMaxPooling(
                kernel_size = 512,
                stride = 512,
                dimension = D),
            )

        # nn.Sequential([stuff])

        self.mlp1 = nn.Sequential(
            ME.MinkowskiLinear(
                64, 256),
            ME.MinkowskiReLU(),
            ME.MinkowskiBatchNorm(256),
            )
        self.mlp2 = nn.Sequential(
            ME.MinkowskiLinear(
                256, 256),
            ME.MinkowskiReLU(),
            ME.MinkowskiBatchNorm(256),
            )
        self.mlp3 = nn.Sequential(
            ME.MinkowskiLinear(
                256, out_feat),
            ME.MinkowskiBatchNorm(out_feat),
            )
        # self.softmax = nn.Sequential(
        #     nn.LogSoftmax(
        #         dim = D)
        #     )
        self.pooling = ME.MinkowskiGlobalPooling() # equiv to maxpoo 512, remove spatial dimension
        # self.linear = ME.MinkowskiLinear(128, out_feat)
        # self.linear = ME.MinkowskiLinear(1, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.mp1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        # out = self.mp2(out)
        out = self.pooling(out)
        out = self.mlp1(out)
        out = self.mlp2(out)
        out = self.mlp3(out)
        
        return out

def train(network, datafile, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum = 0.9)

    n_iter = 10
    
    lossHist = []
    accHist = []
    for labelsPDG, coords, features in tqdm.tqdm(load_batch(datafile, n_iter),
                                                 total = n_iter):

        labels = torch.Tensor([LABELS.index(l) for l in labelsPDG])
        data = ME.SparseTensor(torch.FloatTensor(features).to(device),
                               coordinates=torch.FloatTensor(coords).to(device))

        optimizer.zero_grad()
        outputs = network(data)
        loss = criterion(outputs.F.squeeze(), labels.long())
        loss.backward()
        optimizer.step()

        lossHist.append(loss)

        prediction = torch.argmax(outputs.features, dim = 1)
        accuracy = sum(prediction == labels)/len(prediction)

        accHist.append(accuracy)

    return lossHist, accHist

def trainingPlots(lossHist, accHist, plotDir):
    import os

    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
            
    fig = plt.figure()
    gs = GridSpec(2, 1,
                  figure = fig,
                  height_ratios = [0.5, 0.5],
                  hspace = 0)
    axLoss = fig.add_subplot(gs[0,:])
    axAcc = fig.add_subplot(gs[1,:])
        
    axLoss.plot(lossHist)
    axLoss.axhline(y = -np.log(1./5), ls = '--') # "random guess" loss is -log(0.2)
        
    axAcc.plot(accHist)
        
    axLoss.set_xticklabels([])
    axLoss.set_ylabel('Loss')
    axAcc.set_xlabel('Training iteration')
    axAcc.set_ylabel('Accuracy')
        
    plt.savefig(os.path.join(plotDir,
                             'lossAcc.png'))
