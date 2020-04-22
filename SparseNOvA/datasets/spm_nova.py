'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
import os.path as osp, glob, h5py, tqdm, numpy as np, torch
import utils

class SparsePixelMapNOvA(Dataset):
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
        print(len(self.files))
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
            if counter > nevt: 
#                 print("To check,",counter,"subtract",nevt,"equals")
                counter = counter - nevt
#                 print(counter)
            else:
                event_file = i
                print("The event is",counter,"th event in file",event_file)
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
        data = { 'xview': utils.dense_to_sparse(xview),
                 'yview': utils.dense_to_sparse(yview), 
                 'truth': new_truth}

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
            
