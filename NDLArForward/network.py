import MinkowskiEngine as ME

import torch
import torch.nn as nn
import torch.optim as optim

from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np

from SingleParticleDataAccess import LABELS, BATCH_SIZE, load_batch

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import yaml

import tqdm

import os

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D, manifest):
        super(ExampleNetwork, self).__init__(D)

        self.manifest = manifest

        self.outDir = self.manifest['outdir']
        self.make_output_tree()

        # if there's a checkpoint load it
        if 'checkpoints' in self.manifest:
            latestCheckpoint = self.manifest['checkpoints'][-1]
            self.load_checkpoint(latestCheckpoint)
            self.n_epoch = int(latestCheckpoint.split('_')[-2])
            self.n_iter = int(latestCheckpoint.split('_')[-1].split('.')[0])
            print ("resuming training at epoch {}, iteration {}".format(self.n_epoch, self.n_iter))
        else:
            self.n_epoch = 0
            self.n_iter = 0

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
        self.conv3 = nn.Sequential(
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
                stride = 2,
                dimension = D),
            )
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv5 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=3,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv6 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=3,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())

        self.mp2 = nn.Sequential(
            ME.MinkowskiMaxPooling(
                kernel_size = 2,
                stride = 1,
                dimension = D),
            )
        self.conv7 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=3,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv8 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=3,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv9 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())

        self.mp3 = nn.Sequential(
            ME.MinkowskiMaxPooling(
                kernel_size = 2,
                stride = 1,
                dimension = D),
            )
        self.conv10 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv11 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv12 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        # self.mp3 = nn.Sequential(
        #     ME.MinkowskiMaxPooling(
        #         kernel_size = 512,
        #         stride = 512,
        #         dimension = D),
        #     )

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
        self.pooling = ME.MinkowskiGlobalPooling() # equiv to maxpool 512, remove spatial dimension
        # self.linear = ME.MinkowskiLinear(128, out_feat)
        # self.linear = ME.MinkowskiLinear(1, out_feat)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        out = self.conv3(out)

        # print(out.shape)
        out = self.mp1(out)
        # print(out.shape)
        out = self.conv4(out)
        out = self.conv5(out)
        # print(out.shape)
        out = self.conv6(out)

        # print(out.shape)
        out = self.mp2(out)
        # print(out.shape)
        out = self.conv7(out)
        out = self.conv8(out)
        # print(out.shape)
        out = self.conv9(out)

        # print(out.shape)
        out = self.mp3(out)
        # print(out.shape)
        out = self.conv10(out)
        # print(out.shape)
        out = self.conv11(out)
        out = self.conv12(out)

        # out = self.mp2(out)
        # print(out.shape)
        out = self.pooling(out)
        # print(out.shape)
        out = self.mlp1(out)
        # print(out.shape)
        out = self.mlp2(out)
        # print(out.shape)
        out = self.mlp3(out)
        
        # print(out.shape)
        return out

    def make_checkpoint(self, filename):
        print ("saving checkpoint ", filename)
        torch.save(dict(model = self.state_dict()), filename)

        if not 'checkpoints' in self.manifest:
            self.manifest['checkpoints'] =  []
        self.manifest['checkpoints'].append(filename)

        with open(os.path.join(self.outDir, 'manifest.yaml'), 'w') as mf:
            print ('dumping manifest to', os.path.join(self.outDir, 'manifest.yaml'))
            yaml.dump(self.manifest, mf)

    def load_checkpoint(self, filename):
        print ("loading checkpoint ", filename)
        with open(filename, 'rb') as f:
            checkpoint = torch.load(f)
            print (checkpoint.keys())
            self.load_state_dict(checkpoint['model'], strict=False)

    def make_output_tree(self):
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)
            os.mkdir(os.path.join(self.outDir,
                                  "checkpoints"))
            os.mkdir(os.path.join(self.outDir,
                                  "plots"))
            
        with open(os.path.join(self.outDir, 'manifest.yaml'), 'w') as mf:
            yaml.dump(self.manifest, mf)
            
    def make_plots(self, lossHist, accHist):
        plotDir = os.path.join(self.outDir,
                               "plots")
            
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


    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum = 0.9)

        nEpochs = int(self.manifest['nEpochs'])
        batchesPerEpoch = 400000//BATCH_SIZE
       
        report = False
        prevRemainder = 0
    
        lossHist = []
        accHist = []
        for i in tqdm.tqdm(range(nEpochs)):
            if i < self.n_epoch:
                continue
            for j, (labelsPDG, 
                    coords, 
                    features) in tqdm.tqdm(enumerate(load_batch(self.manifest['trainfile'],
                                                                n_iter = batchesPerEpoch)),
                                           total = batchesPerEpoch):
                if j < self.n_iter:
                    continue

                labels = torch.Tensor([LABELS.index(l) for l in labelsPDG]).to(device)
                data = ME.SparseTensor(torch.FloatTensor(features).to(device),
                                       coordinates=torch.FloatTensor(coords).to(device))

                optimizer.zero_grad()

                if report:
                    with profile(activities=[ProfilerActivity.CUDA],
                                 profile_memory = True,
                                 record_shapes = True) as prof:
                        with record_function("model_inference"):
                            outputs = self(data)

                    print(prof.key_averages().table(sort_by="self_cuda_time_total", 
                                                    row_limit = 10))
                    
                else:
                    outputs = self(data)

                loss = criterion(outputs.F.squeeze(), labels.long())
                loss.backward()
                optimizer.step()
        
                self.n_iter += 1

                # save a checkpoint of the model every 10% of an epoch
                remainder = (self.n_iter/batchesPerEpoch)%0.1
                if remainder < prevRemainder:
                    try:
                        checkpointFile = os.path.join(self.outDir,
                                                      'checkpoints',
                                                      'checkpoint_'+str(self.n_epoch)+'_'+str(self.n_iter)+'.ckpt')
                        self.make_checkpoint(checkpointFile)
                    except AttributeError:
                        pass
                prevRemainder = remainder
            
            self.n_epoch += 1
            senf.n_iter = 0

            lossHist.append(float(loss))
        
            prediction = torch.argmax(outputs.features, dim = 1)
            accuracy = sum(prediction == labels)/len(prediction)

            accHist.append(float(accuracy))

        return lossHist, accHist
