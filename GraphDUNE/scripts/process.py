#!/usr/bin/env python

if __name__ == "__main__":
  import sys, os, os.path as osp
  for p in [ "/scratch", "/numl"]:
    if p not in sys.path: sys.path.append(p)
  import argparse, numl, glob, multiprocessing as mp
  from numl.core.out import *
  from numl.graph.edges import *
  from functools import partial

  args = argparse.ArgumentParser()
  args.add_argument("-i", "--input", type=str, required=True,
                    help="input directory to read files from")
  args.add_argument("-o", "--output", type=str, required=True,
                    help="output directory to write files to")
  args.add_argument("-p", "--processes", type=int, default=None,
                    help="number of threads to parallelise over")
  args.add_argument("-h5", "--hdf5", default=False, action="store_true",
                    help="write output to HDF5 file instead of PyTorch files")
  opts = args.parse_args()

  files = glob.glob("{}/*.h5".format(opts.input))
  if not osp.exists(opts.output): os.mkdir(opts.output)

  if opts.hdf5: out = H5Out(osp.join(opts.output, "out.h5"))
  else: out = PTOut(opts.output)

  kwargs = {
    "out": out,
    "g": numl.process.hitgraph.process_event,
    # "g": numl.process.hitgraph.process_event_singleplane,
    "l": numl.labels.standard.panoptic_label,
    # "l": numl.labels.ccqe.semantic_label,
    "e": numl.graph.edges.delaunay,
    # "e": numl.graph.edges.radius,
  }

  if opts.processes is not None:
    from functools import partial
    func = partial(numl.process.hitgraph.process_file, **kwargs)
    import multiprocessing as mp
    with mp.Pool(processes=opts.processes) as pool:
      pool.map(func, files)
  else:
    numl.process.hitgraph.process_file(files[1], **kwargs)
    # for f in files: numl.process.hitgraph.process_file(f, **kwargs

  print("done processing files!")

