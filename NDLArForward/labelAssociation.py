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

from SingleParticleDataAccess import LABELS, load_batch

import tqdm

import yaml
import os
            
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    net = ExampleNetwork(in_feat=1, out_feat=5, D=3, manifest = manifest).to(device)

    if args.force:
        # remove previous checkpoints
        for oldCheckpoint in os.listdir(os.path.join(manifest['outdir'],
                                                     'checkpoints')):
            print ('removing ', oldCheckpoint)
            os.remove(os.path.join(manifest['outdir'],
                                   'checkpoints',
                                   oldCheckpoint))
        net.manifest['checkpoints'] = []
        reportFile = os.path.join(manifest['outdir'],
                                  'train_report.dat')
        if os.path.exists(reportFile):
            os.remove(reportFile)
        
    net.train()

    checkpointFile = os.path.join(net.outDir,
                                  'checkpoint_final_{}_{}.ckpt'.format(manifest['nEpochs'], 0))
    net.make_checkpoint(checkpointFile)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', type = str,
                        default = "/home/dan/studies/NDLArForwardME/testManifest.yaml",
                        help = "network manifest yaml file")
    parser.add_argument('-f', '--force',
                        action = 'store_true',
                        help = "forcibly train the network from scratch")
    parser.add_argument('-v', '--verbose',
                        action = 'store_true',
                        help = "print extra debug messages")
    
    
    args = parser.parse_args()
    
    main(args)
