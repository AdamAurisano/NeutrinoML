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

        self.files = sorted(glob.glob(f'{self.filedir}/*.h5'))
        
        # See if we can load metadata from file
        meta = f'{self.filedir}/metadata.pt'
        if osp.exists(meta):
            self.file_metadata = torch.load(meta)
            # Validate that the metadata looks good
            if len(self.file_metadata) != len(self.files):
                raise Exception('Mismatch between files and metadata.')
        # If there's no metadata file, then generate metadata
        else:
            self.get_file_metadata()
            torch.save(self.file_metadata, meta)

        # Add up the total event count
        self.total_events = 0
        for val in self.file_metadata:
            self.total_events += len(val['mask'])

        # print(self.total_events)
        # print(self.file_metadata[2]['mask'])

        # Here we can do a glob to get a list of all file names
        # We can then open each of those files and count the number of images
        # to count up how many training images we have in total
        #
        # In python you can set any object as a class variable, meaning
        # it will not be discarded at the end of the function call, and
        # will remain as long as the object exists. So if I did
        #
        #      n_files = 0
        #
        # then n_files will go out of scope and be lost when the __init__
        # function returns. However, if I instead do
        #
        #      self.n_files = 0
        #
        # then n_files will be assigned to the dataset object, and
        # persists after __init__ returns. If we initialise a SparsePixelMap
        # object, we can do
        #
        #      dataset = SparsePixelMapNOvA(filedir=/path/to/files)
        #      print(dataset.n_files)
        #
        # By declaring n_files with the "self." prefix, we've declared it as
        # what's called a "data attribute" of the class - an instance of this
        # class will hold onto it for as long as that instance exists
        
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
#         print(j)
        image = self.current_file[1]['rec.training.slicemaps']['slicemap'][j]
        xview, yview = image.reshape(2, 448, 384)[:]
        data = { 'xview': utils.dense_to_sparse(xview),
                 'yview': utils.dense_to_sparse(yview) }

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

    #         # situation 1 to open/close file
    #         if self.current_file[0] is not None and self.current_file[0] != event_file:
    #             self.current_file[1].close()
    #             self.current_file = (event_file, h5py.File(self.files[event_file]))
    #             self.current_file.open()
    #             # Check to see if there is an open file beforehand.

    #         # situation 2 to open/close file
    #         else:
    #             # situation 3 to open/close file
    #             if self.current_file[0] is None:
    #                 self.current_file = (event_file, h5py.File(self.files[event_file]))
    #                 self.current_file.open()
    #             # situation 4 to open/close file
    #             elif self.current_file[0] == event_file:
    #                 pass
              
    # This is where most of the work needs to be done. Say we have 10 files,
    # each containing 10,000 images – 100,000 images total.
    # This function takes in an integer in the 0-99999 range (and should throw
    # an error if it's outside that range). The function needs to know how many
    # files there are, and how many images each one contains. It needs to know
    # that image number 15,000 is actually image 5,000 in file 2, and then
    # open up that file and extract the corresponding pixel map.
    #
    # It should minimise the opening and closing of files. When we open a file,
    # we can hold onto it and only close it when necessary. For file i, we can
    # do
    #
    # self.current_file = (i, h5py.File(self.files[i]))
    #
    # to open the file, and keep hold of it. current_file is a tuple - basically
    # a two-element array, where current_file[0] is the file number and
    # current_file[1] holds the file itself. Then the next time we load an image
    # for file i:
    #
    # if (i != self.current_file[0]):
    #   self.current_file.close()
    #   self.current_file = (i, h5py.File(self.files[i]))
    #
    # so if file i is already open, we skip the open/close part and go straight to
    # loading the image.
    #
    # The first step is just getting the bookkeeping right here. Once that's
    # done, we can add in a few more minor things like transforming the
    # pixel maps into PyTorch tensors in a sparse format.
