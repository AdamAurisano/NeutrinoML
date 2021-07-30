#!/usr/bin/env python

if __name__ == "__main__":
  import argparse, numl, glob
  args = argparse.ArgumentParser(description="preprocess HDF5 files into ML inputs")
  args.add_argument("-i", "--input-dir", type=str, required=True,
                    help="directory containing input files to process")
  args.add_argument("-o", "--output-dir", type=str, required=True,
                    help="directory to write output files to")
  opts = args.parse_args()
  out = numl.core.out.PTOut(opts.output_dir)
  for f in glob.glob("{}/*.h5".format(opts.input_dir)):
    numl.process.spmap.process_file(out, f)

