import torch
import torch.nn as nn
import MinkowskiEngine as ME

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=1,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=D),
            # ME.MinkowskiBatchNorm(1),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                dimension=D),
            # ME.MinkowskiBatchNorm(1),
            ME.MinkowskiReLU())
        self.mp1 = nn.Sequential(
            ME.MinkowskiMaxPooling(
                kernel_size = 2,
                stride = 1,
                dimension = D),
            )
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                dimension=D),
            # ME.MinkowskiBatchNorm(1),
            ME.MinkowskiReLU())
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                dimension=D),
            # ME.MinkowskiBatchNorm(1),
            ME.MinkowskiReLU())
        self.mp2 = nn.Sequential(
            ME.MinkowskiMaxPooling(
                kernel_size = 512,
                stride = 1,
                dimension = D),
            )
        # self.mlp1 = nn.Sequential(
        #     nn.MLP(
        #         in_channels = 1,
        #         hidden_channels = [1, 1],
        #         norm_layer = 5,
        self.pooling = ME.MinkowskiGlobalPooling()
        # self.linear = ME.MinkowskiLinear(128, out_feat)
        self.linear = ME.MinkowskiLinear(1, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.mp1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.mp2(out)
        out = self.pooling(out)
        return self.linear(out)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = ExampleNetwork(in_feat=1, out_feat=1, D=3).to(device)

    from SingleParticleDataAccess import load_batch
    labels, coords, features = next(load_batch(args.infile))

    coords = coords[:10,:]
    features = features[:10,:]
    
    data = ME.SparseTensor(torch.FloatTensor(features).to(device),
                           coordinates=torch.FloatTensor(coords).to(device))

    print (data.shape)
    
    # # Forward
    # output = net(data)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type = str,
                        default = "/home/dan/studies/NDLArForwardME/train.root",
                        help = "input  file")

    args = parser.parse_args()
    
    main(args)
