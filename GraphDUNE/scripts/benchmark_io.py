#!/usr/bin/env python

if __name__ == "__main__":
  import sys, os, os.path as osp
  for p in [ "/scratch", "/numl"]:
    if p not in sys.path: sys.path.append(p)
  import argparse, numl, glob, multiprocessing as mp
  from numl.core.out import *
  from numl.graph.edges import *

  import time
  files = glob.glob("/data/uboone/pandora/hdf5/*.h5")

  def test1(fname):

    t1 = time.time()

    f = numl.core.file.NuMLFile2(fname)
    evt =  f.get_dataframe("event_table",    ["event_id"])
    part = f.get_dataframe("particle_table", ["event_id","g4_id","parent_id","type","start_process","end_process"])
    hit =  f.get_dataframe("hit_table")
    edep = f.get_dataframe("edep_table")
    
    t2 = time.time()
    print(f"loading dataframes took {t2-t1:.2f}s")

    import psutil
    print(psutil.Process().memory_info().rss / (1024 * 1024))

    import tqdm
    for i in tqdm.tqdm(range(len(f))):
      index = evt.index[i]
      if index not in part.index or index not in hit.index or index not in edep.index: continue
      evt_part = part.loc[index].copy().reset_index(drop=True)
      evt_hit = hit.loc[index].copy().reset_index(drop=True)
      evt_edep = edep.loc[index].copy().reset_index(drop=True)

    t3 = time.time()
    print(f"slicing events took {t3-t2:.2f}s")

  def test2(fname):

    t1 = time.time()

    f = numl.core.file.NuMLFile(fname)
    f.add_group("particle_table", ["event_id","g4_id","parent_id","type","start_process","end_process"])
    f.add_group("hit_table")
    f.add_group("edep_table")

    t2 = time.time()
    print(f"loading file took {t2-t1:.2f}s")

    import tqdm
    for i in tqdm.tqdm(range(len(f))):
      evt = f[i]
      print(evt)

    t3 = time.time()
    print(f"slicing events took {t3-t2:.2f}s")

  test1(files[0])
  print(f"max memory usage was {max(mem)}")

  test2(files[0])
  print(f"max memory usage was {max(mem)}")

