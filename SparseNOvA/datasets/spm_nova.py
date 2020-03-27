'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
from glob import glob

class SparsePixelMapNOvA(Dataset):
  def __init__(self, filedir, **kwargs):
    self.filedir = filedir
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
    return 1
    # This function needs to return the number of images in the dataset

  def __getitem__(self, idx):
    return 1
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
