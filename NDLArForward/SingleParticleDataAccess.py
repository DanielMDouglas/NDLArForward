# from larcv import larcv
import larcv
import numpy as np
import plotly
import plotly.graph_objs as go
import sys, yaml
# set software directory
software_dir = 'lartpc_mlreco3d'
sys.path.insert(0,software_dir)

def cfg_mlreco(data_key):
    cfg = """
iotool:
  batch_size: 1
  shuffle: False
  num_workers: 1
  collate_fn: CollateSparse
  sampler:
    name: RandomSequenceSampler
  dataset:
    name: LArCVDataset
    data_keys:
      - """+data_key+"""
    limit_num_files: 10
    schema:
      data:
        - parse_sparse3d
        - sparse3d_pcluster
      particle:
        - parse_particle_asis
        - particle_corrected
        - cluster3d_pcluster
    """

    from mlreco.main_funcs import process_config, prepare
    cfg_dict=yaml.load(cfg,Loader=yaml.Loader)
    process_config(cfg_dict)
    hs=prepare(cfg_dict)

    yield next(hs.data_io_iter)

def get_labels(parts):
    LABELS = [11,22,13,211,2212]
    pdg = 0
    results=[]
    for event in parts:
        pdg = 0
        for p in event:
            if not p.track_id() == p.parent_track_id():
                continue
            pdg = p.pdg_code()
            break
        if pdg == 0:
            print('Primary particle not found')
            raise ValueError
        if not pdg in LABELS:
            print('Unexpected PDG for a primary',pdg)
            raise ValueError
        LABELS.index(pdg)
        results.append(pdg)
    return np.array(results)

def load_batch(data_file):
    batch = next(cfg_mlreco(data_file))
    
    labels = get_labels(batch['particle'])
    coords = batch['data'][:,0:4]
    features = np.array([batch['data'][:,4]]).T

    yield labels, coords, features
