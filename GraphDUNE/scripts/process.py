#!/usr/bin/env python

import sys, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import argparse, numl

def main():

  args = argparse.ArgumentParser()
  args.add_argument("-i", "--infile", type=str, required=True,
                    help="input HDF5 file")
  args.add_argument("-o", "--outfile", type=str, required=True,
                    help="output file name")
  args.add_argument("-5", "--hdf5", action="store_true",
                    help="write output to HDF5 files")
  args.add_argument("-p", "--profile", action="store_true",
                    help="run timing profile")
  args.add_argument("-s", "--sequence", action="store_true",
                    help="use sequencing")
  opts = args.parse_args()

  out = numl.core.out.H5Out(opts.outfile) if opts.hdf5 else numl.core.out.PTOut(opts.outfile)
  numl.process.hitgraph.process_file(out, opts.infile, l=numl.labels.standard.panoptic_label, \
                                     use_seq=opts.sequence, profile=opts.profile)

if __name__ == "__main__":
   main()

