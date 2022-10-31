from larcv import larcv
import numpy as np
import plotly
import plotly.graph_objs as go
import sys, yaml
# set software directory
software_dir = 'lartpc_mlreco3d'
sys.path.insert(0,software_dir)

cfg = """
iotool:
  batch_size: 32
  shuffle: False
  num_workers: 4
  collate_fn: CollateSparse
  sampler:
    name: RandomSequenceSampler
  dataset:
    name: LArCVDataset
    data_keys:
      - /sdf/group/neutrino/kterao/singlep_uq/data/train.root
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
# pre-process configuration (checks + certain non-specified default settings)
process_config(cfg_dict)
# prepare function configures necessary "handlers"
hs=prepare(cfg_dict)

batch = next(hs.data_io_iter)

print('Loaded keys:',batch.keys())

# The label count and sample count in input should match (to batch size in the config)
data = batch['data']
print('Got',len(data),'voxels in this batch')
print('Got',len(np.unique(data[:,0])),'samples in this batch')

print('length of index data',len(batch['index']))
print(batch['index'])

# Plot energy depositions (input data)
import plotly
import plotly.graph_objs as go
colors = plotly.colors.qualitative.Light24

layout = go.Layout(
showlegend=True,
legend=dict(x=0.95,y=0.95,xanchor='right',yanchor='top'),
width=1024,
height=768,
hovermode='closest',
margin=dict(l=0,r=0,b=0,t=0),                                                                                                                                  
template='plotly_dark',                                                                                                                                        
uirevision = 'same',

)

def plot(batch,entry):
    
    entry = int(entry)
    
    image = batch['data']      # Access voxel information
    parts = batch['particle']  # Access particle information
    index = batch['index']     # Access sample index
    
    # Check if the specified entry is valid
    if not entry in np.unique(image[:,0]):
        print('ERROR: the entry %d does not exist in this batch data' % entry)
        return
    
    # Find the primary particle in this image to find out its PDG for a label
    pdg = 0
    for p in parts[entry]:
        if not p.track_id() == p.parent_track_id():
            continue
        pdg = p.pdg_code()
        break
    # Make sure the primary particle is found
    if pdg == 0:
        print('ERROR: cannot find the primary particle for entry %d' % entry)
        return
    
    # Slice a tensor for the specified entry and create a trace to visualize
    image = image[np.where(image[:,0]==entry)]
    trace = go.Scatter3d(x=image[:,1],y=image[:,2],z=image[:,3],
                         mode='markers',
                         name='Entry %d ... PDG %d' % (index[entry],pdg),
                         marker=dict(size=2,color=image[:,4],opacity=0.3),
                         hovertext=['%.2f [MeV]' % pt[4] for pt in image],
                        )
    fig = go.Figure(data=trace,layout=layout)
    fig.show()
    
plot(batch,1)
