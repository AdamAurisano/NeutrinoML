'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
import sys, os.path as osp, yaml, argparse, logging, math, numpy as np, torch, sherpa, logging
import matplotlib.pyplot as plt
sys.path.append('/scratch')
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from Core.trainers import Trainer
from glob import glob
from Core import utils
from SparseNOvA import datasets
from Core import models

def __init__(self, filedir, **kwargs):
    '''Initialiser for SparsePixelMapNOvA class'''
    self.filedir = filedir
    self.current_file = None

    nonswap = sorted(glob.glob(f'{self.filedir}/nonswap/*.h5'))
    fluxswap = sorted(glob.glob(f'{self.filedir}/fluxswap/*.h5'))
    tauswap = sorted(glob.glob(f'{self.filedir}/tauswap/*.h5'))

    # We want to sort the files in a consistent way
    # but also have a mix of all file types throughout
    self.files = list()
    for triplet in zip(nonswap, fluxswap, tauswap): self.files += triplet
    self.file_metadata = list()
    for file in self.files:
        base = osp.basename(file).split('.')[0]
        metafile = f'{self.filedir}/metadata/{base}.pt'
        self.file_metadata.append(torch.load(metafile))

    # Add up the total event count
    self.total_events = 0
    for val in self.file_metadata:
        self.total_events += len(val['mask'])

def __len__(self):
    return self.total_events

def __getitem__(self, idx):
    '''Return training information at provided index'''
    if not 0 <= idx < self.total_events:
        raise Exception(f'Event number {idx} invalid – must be in range 0 -> {self.total_events-1}.')

    # First step: find the file the neutrino event is in.
    counter = idx
    event_number = 0
    event_file = 0
    for i in range(len(self.file_metadata)):
        nevt = len(self.file_metadata[i]['mask'])
        if counter >= nevt: counter -= nevt
        else:
            event_file = i
            break

    # Second step: figure out if the file is open/close
    if self.current_file is None:
        self.current_file = (event_file, h5py.File(self.files[event_file], 'r'))
    elif self.current_file[0] != event_file:
        self.current_file[1].close()
        self.current_file = (event_file, h5py.File(self.files[event_file], 'r'))

    # Third step: once the correct file is opened, pull the image out
    # i tells you the file
    # j tells you the position in the file
    j = self.file_metadata[i]['mask'][counter]
    truth = self.file_metadata[i]['truth'][counter]
    new_truth = self.get_alias(truth)
    image = self.current_file[1]['rec.training.slicemaps']['slicemap'][j]
    xview, yview = image.reshape(2, 448, 384)[:]
    xsparse = torch.tensor(xview).float().to_sparse()
    print(xsparse[0])
    ysparse = torch.tensor(yview).float().to_sparse()
    data = { 'xfeats': xsparse._values().unsqueeze(dim=-1),
             'xcoords': xsparse._indices().T.int(),
             'yfeats': ysparse._values().unsqueeze(dim=-1),
             'ycoords': ysparse._indices().T.int(),
             'truth': torch.tensor(new_truth).long() }
    print(data['xfeats'])
    # n by 1 dimension
    return data


def get_file_metadata(self):
    '''Function to produce metadata for all training files and return a dictionary'''
    self.file_metadata = []
    for filename in tqdm.tqdm(self.files):
        file = h5py.File(filename, 'r')
        mask = np.nonzero(file['rec.mc']['nnu'][:,0]==1)[0]
        good = np.ones(mask.shape[0], dtype=bool) # Use this mask to kill the images with no associated truth
        truth = []
        # Loop over each neutrino image to look for the associated truth
        for i, nu in enumerate(mask):
            true_idx = self.match_truth(file, nu)
            # Remove this image if there's no associated truth
            if len(true_idx) == 0: good[i] = False
            # Otherwise get the truth and add it to the metadata
            else: truth.append(file['rec.training.trainingdata']['interaction'][true_idx,0][0])
        mask = mask[good]
        if (len(mask) != len(truth)):
            raise Exception(f'Mismatch found: {len(mask)} images and {len(truth)} truths.')
        self.file_metadata.append({ 'mask': mask, 'truth': truth })
        file.close()
    return

def match_truth(self, file, im_idx):
    '''Returns list of index locations of trainingdata for a given slicemap'''
    keys = ['run','subrun','cycle','evt','subevt']
    tag = { key: file['rec.training.slicemaps'][key][im_idx,0] for key in keys }
    mask = None
    for key in keys:
        m = (file['rec.training.trainingdata'][key][:,0] == tag[key])
        mask = m if mask is None else mask & m
    true_idx = mask.nonzero()[0]
    if len(true_idx) > 1:
        raise Exception(f'{len(true_idx)} truths found for slicemap {im_idx}.')
    return true_idx

def get_alias(self, flav):
    '''Function that alias interaction enum to the following list'''
    # [0 == nu_mu, 1 == nu_e, 2 == nu_tau, 3 == NC, 4 == others]
    if 0 <= flav < 4:
        flav = 0
    elif 4 <= flav < 8:
        flav = 1
    elif 8 <= flav < 12:
        flav = 2
    elif flav == 13:
        flav = 3
    else:
        flav = 4
#         maybe fifth class
    return flav

def main():
    parser = argparse.ArgumentParser('energy_density.py')
    parser.add_argument('config', nargs='?', default='/scratch/SparseNOvA/config/nova_sparse_fishnet.yaml')
    with open(parser.parse_args().config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    all_nus = sorted(glob(f'{config["data"]["filedir"]}/nu/*.pt'))
    all_cosmics = sorted(glob(f'{config["data"]["filedir"]}/cosmic/*.pt'))

    if len(all_cosmics) > int(0.1 * len(all_nus)):
        all_cosmics = all_cosmics[0:int(0.1*len(all_nus))]

    fulllen_nu = len(all_nus)
    fulllen_cosmic = len(all_cosmics)

    tv_num_nu = math.ceil(fulllen_nu*config['data']['t_v_split'])
    tv_num_cosmic = math.ceil(fulllen_cosmic*config['data']['t_v_split'])

    splits_nu = np.cumsum([fulllen_nu - tv_num_nu, 0, tv_num_nu])
    splits_cos = np.cumsum([fulllen_cosmic - tv_num_cosmic, 0, tv_num_cosmic])

    train_files = all_nus[0:splits_nu[1]] + all_cosmics[0:splits_cos[1]]
    train_files.sort(key = lambda x: osp.basename(x))  
    train_dataset = datasets.get_dataset(filelist=train_files, apply_jitter=True, normalize_coord=True, **config['data'])

    valid_files = all_nus[splits_nu[1]:splits_nu[2]] + all_cosmics[splits_cos[1]:splits_cos[2]]
    valid_files.sort(key = lambda x: osp.basename(x))
    valid_dataset = datasets.get_dataset(filelist=valid_files, apply_jitter=False, normalize_coord=True, **config['data'])
    
    energy = []
    for i in range(len(train_dataset)):
        xfeats = (train_dataset[i]['xfeats'].squeeze()).tolist()
        energy.extend(xfeats)
        if i == 100:
            break
    
    n, bins, patches = plt.hist(energy, bins=100, density=True, cumulative=True, histtype='step')

    plt.title('Neutrino Energy Cumulative Distribution')
    plt.xlabel('Neutrino Energy')
    plt.ylabel('Probbability Density')
    plt.savefig('cdf_energies.png', bbox_inches='tight')
    plt.close()
    
    
if __name__ == '__main__':
  main()
