import sys, os.path as osp
for p in [ "/scratch", "/numl"]:
  if p not in sys.path: sys.path.append(p)
import numl, glob, multiprocessing as mp
from functools import partial

files = glob.glob("/data/pandora/hdf5/*")
func = partial(numl.process.hitgraph.process_file, "/data/pandora/processed")
with mp.Pool(processes=50) as pool: pool.map(func, files)

