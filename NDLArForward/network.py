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

        # save the manifest dict internally
        self.manifest = manifest

        # make the output data structure
        self.outDir = self.manifest['outdir']
        self.reportFile = os.path.join(self.manifest['outdir'],
                                       'train_report.dat')
        self.make_output_tree()

        self.n_epoch = 0
        self.n_iter = 0

        # load layer structure from the manifest
        self.layers = []
        layer_in_feat = in_feat
        for layer in self.manifest['layers']:
            if layer['type'] == 'MConvolution':
                layer_out_feat = int(layer['out_feat'])
                self.layers.append(ME.MinkowskiConvolution(
                    in_channels = layer_in_feat,
                    out_channels = layer_out_feat,
                    kernel_size = int(layer['kernel_size']),
                    stride = int(layer['stride']),
                    bias = False,
                    dimension = D
                ))
                layer_in_feat = layer_out_feat
            elif layer['type'] == 'MReLU':
                self.layers.append(ME.MinkowskiReLU())
            elif layer['type'] == 'MBatchNorm':
                self.layers.append(ME.MinkowskiBatchNorm(layer_out_feat))
            elif layer['type'] == 'MMaxPooling':
                self.layers.append(ME.MinkowskiMaxPooling(
                    kernel_size = int(layer['kernel_size']),
                    stride = int(layer['stride']),
                    dimension = D
                ))
            elif layer['type'] == 'MLinear':
                layer_out_feat = int(layer['out_feat'])
                self.layers.append(ME.MinkowskiLinear(
                    layer_in_feat,
                    layer_out_feat
                ))
                layer_in_feat = layer_out_feat
            elif layer['type'] == 'MGlobalPooling':
                self.layers.append(ME.MinkowskiGlobalPooling())

        self.network = nn.Sequential(*self.layers)
            
    def forward(self, x):
        return self.network(x)

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
        else:
            self.manifest['checkpoints'] = []
            for existingCheckpoint in os.listdir(os.path.join(self.outDir,
                                                              "checkpoints")):
                fullPath = os.path.join(self.outDir,
                                        "checkpoints",
                                        existingCheckpoint)
                self.manifest['checkpoints'].append(fullPath)
            self.manifest['checkpoints'].sort(key = lambda name: int(name.split('_')[-2]) + \
                                              int(name.split('_')[-1].split('.')[0])*0.001)

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
        """
        page through a training file, do forward calculation, evaluate loss, and backpropagate
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum = 0.9)

        nEpochs = int(self.manifest['nEpochs'])
        batchesPerEpoch = 400000//BATCH_SIZE
       
        report = False
        prevRemainder = 0

        # if there's a previous checkpoint, start there
        if 'checkpoints' in self.manifest and self.manifest['checkpoints'] != []:
            latestCheckpoint = self.manifest['checkpoints'][-1]
            self.load_checkpoint(latestCheckpoint)
            self.n_epoch = int(latestCheckpoint.split('_')[-2])
            self.n_iter = int(latestCheckpoint.split('_')[-1].split('.')[0])
            print ("resuming training at epoch {}, iteration {}".format(self.n_epoch, self.n_iter))
    
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

                        prediction = torch.argmax(outputs.features, dim = 1)
                        accuracy = sum(prediction == labels)/len(prediction)

                        self.training_report(loss, accuracy)

                        device.empty_cache()
                    except AttributeError:
                        pass
                prevRemainder = remainder
            
            self.n_epoch += 1
            self.n_iter = 0

    def training_report(self, loss, acc):
        """
        Add to the running report file at a certain moment in the training process
        """

        with open(self.reportFile, 'a') as rf:
            rf.write('{} \t {} \t {} \t {} \n'.format(self.n_epoch, 
                                                      self.n_iter, 
                                                      loss, 
                                                      acc))
        

    def evaluate(self):
        """
        page through a test file, do forward calculation, evaluate loss and accuracy metrics
        do not update the model!
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        evalBatches = 50
        # evalBatches = 10
       
        report = False
        # report = True
        
        criterion = nn.CrossEntropyLoss()

        lossList = []
        accList = []

        for (labelsPDG, 
             coords, 
             features) in tqdm.tqdm(load_batch(self.manifest['testfile'],
                                               n_iter = evalBatches),
                                    total = evalBatches):
            
            labels = torch.Tensor([LABELS.index(l) for l in labelsPDG]).to(device)
            data = ME.SparseTensor(torch.FloatTensor(features).to(device),
                                   coordinates=torch.FloatTensor(coords).to(device))

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
            
            self.n_iter += 1

            lossList.append(float(loss))
        
            prediction = torch.argmax(outputs.features, dim = 1)
            accuracy = sum(prediction == labels)/len(prediction)

            accList.append(float(accuracy))

            # if not self.n_iter % 10:
            device.empty_cache()

        return lossList, accList
