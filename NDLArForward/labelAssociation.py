import torch
torch.manual_seed(12)

import random
random.seed(12)

import numpy as np
np.random.seed(12)

import torch.nn as nn
import torch.optim as optim

import MinkowskiEngine as ME

from network import ExampleNetwork, train, trainingPlots

from SingleParticleDataAccess import LABELS, load_batch

import tqdm
            
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = ExampleNetwork(in_feat=1, out_feat=5, D=3).to(device)

    if args.checkpoint:
        with open(args.checkpoint, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)
            net.load_state_dict(checkpoint['model'],
                                strict=False)

    loss, acc = train(net, args.infile, plotDir = args.plots)

    if args.plots:
        trainingPlots(loss, acc, args.plots)
    
    if args.output:
        torch.save(dict(model = net.state_dict()), args.output)

    net
    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type = str,
                        default = "/home/dan/studies/NDLArForwardME/train.root",
                        help = "input file")
    parser.add_argument('-o', '--output', type = str,
                        default = "weights-01000.ckpt",
                        help = "save the model checkpoint")
    parser.add_argument('-c', '--checkpoint', type = str,
                        help = "begin from a saved checkpoint")
    parser.add_argument('-p', '--plots', type = str,
                        help = "write the plots to this directory")
    
    args = parser.parse_args()
    
    main(args)
