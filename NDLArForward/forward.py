import torch
import torch.nn as nn
import MinkowskiEngine as ME

from LarpixParser import event_parser as EvtParser
from LarpixParser import hit_parser as HitParser
from LarpixParser.geom_to_dict import multi_layout_to_dict_nopickle
from LarpixParser import util as util

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)

def load_hits(filename, event_id, configs, device):
    f = h5py.File(filename, 'r')
    packets = f['packets']
 
    geom_dict = multi_layout_to_dict_nopickle(configs['pixel'],
                                              configs['detprop'])
    run_config = util.get_run_config(configs['detprop'])

    detector.set_detector_properties(configs['detprop'],
                                     configs['pixel'])
    
    t0_grp = EvtParser.get_t0(packets)

    t0 = t0_grp[event_id][0]
    print("--------event_id: ", event_id)
    pckt_mask = (packets['timestamp'] > t0) & (packets['timestamp'] < t0 + 3330)
    # pckt_mask = (packets['timestamp'] > t0)
    packets_ev = packets[pckt_mask]
        
    x,y,z,dQ = HitParser.hit_parser_charge(t0, packets_ev, geom_dict, run_config)

    coords, feats = ME.utils.sparse_collate(feats = [torch.FloatTensor(dQ).to(device)],
                                            coords = [torch.FloatTensor([x, y, z]).to(device)])

    return MD.SparseTensor(feats, coordinates=coords)
                                                     
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = ExampleNetwork(in_feat=1, out_feat=5, D=2).to(device)
    #print(net)

    data=load_hits(args.infile,
                   args.eventid,
                   {'pixel': args.pixelfile,
                    'detprop': args.detprop},
                   device=device)

    # Forward
    output = net(data)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type = str,
                        help = "input LArPix file")
    parser.add_argument('-e', '--eventid', type = int,
                        default = 0,
                        help = "geometry layout pickle file")
    parser.add_argument('-p', '--pixelfile', type = str,
                        default = "../../larpix_readout_parser/config_repo/multi_tile_layout-3.0.40.yaml",
                        help = "pixel layout yaml file")
    parser.add_argument('-d', '--detprop', type = str,
                        default = "../../larpix_readout_parser/config_repo/ndlar-module.yaml",
                        help = "detector properties yaml file")

    args = parser.parse_args()

    main(args)
