import torch
# torch.manual_seed(12)

import random
# random.seed(12)

import numpy as np
# np.random.seed(12)

import torch.nn as nn
import torch.optim as optim

import MinkowskiEngine as ME

from network import ExampleNetwork

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from SingleParticleDataAccess import LABELS, load_batch

import tqdm

import yaml
import os
            
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    net = ExampleNetwork(in_feat=1, out_feat=5, D=3, manifest = manifest).to(device)

    epochs = np.unique([int(checkpoint.split('_')[-2]) for checkpoint in manifest['checkpoints']])

    lastCheckpoints = []

    for thisEpoch in epochs:
        theseCheckpoints = []
        for checkpoint in manifest['checkpoints']:
            n_epoch = int(checkpoint.split('_')[-2])
            if thisEpoch == n_epoch:
                theseCheckpoints.append(checkpoint)

        lastCheckpoints.append(theseCheckpoints[-1])

    print ("last checkpoints: ", lastCheckpoints)

    meanLoss = []
    errLoss = []

    meanAcc = []
    errAcc = []

    epoch = []

    for e, checkpoint in enumerate(lastCheckpoints):
        net.load_checkpoint(checkpoint)
        loss, acc = net.evaluate()

        epoch.append(e)

        meanLoss.append(np.mean(loss))
        errLoss.append(np.quantile(loss, (0.16, 0.84)))

        meanAcc.append(np.mean(acc))
        errAcc.append(np.quantile(acc, (0.16, 0.84)))

    plotDir = os.path.join(manifest['outdir'],
                           "plots")
            
    fig = plt.figure()
    gs = GridSpec(2, 1,
                  figure = fig,
                  height_ratios = [0.5, 0.5],
                  hspace = 0)
    axLoss = fig.add_subplot(gs[0,:])
    axAcc = fig.add_subplot(gs[1,:])
        
    errLoss = np.abs(np.array(errLoss).T - np.array(meanLoss))
    print ("shape: ", np.array(errLoss).shape)
    axLoss.errorbar(epoch, meanLoss, 
                    yerr = errLoss, 
                    fmt = 'o')
    axLoss.axhline(y = -np.log(1./5), 
                   ls = '--') # "random guess" loss is -log(0.2)
        
    errAcc = np.abs(np.array(errAcc).T - np.array(meanAcc))
    axAcc.errorbar(epoch, meanAcc, 
                   yerr = errAcc, 
                   fmt = 'o')
    
    axLoss.set_xticklabels([])
    axLoss.set_ylabel('Loss')
    axAcc.set_xlabel('Epoch')
    axAcc.set_ylabel('Accuracy')
        
    plt.savefig(os.path.join(plotDir,
                             'lossAcc.png'))
    
    outArray = np.ndarray((7, len(epoch)))
    outArray[0,:] = epoch
    outArray[1,:] = meanLoss
    outArray[2,:] = errLoss[0,:]
    outArray[3,:] = errLoss[1,:]
    outArray[4,:] = meanAcc
    outArray[5,:] = errAcc[0,:]
    outArray[6,:] = errAcc[1,:]

    np.savetxt(os.path.join(manifest['outdir'],
                            "testEval.dat"),
               outArray)

    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/home/dan/studies/NDLArForwardME/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    
    
    args = parser.parse_args()
    
    main(args)
