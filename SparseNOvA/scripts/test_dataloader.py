#!/usr/bin/env python

import sys
sys.path.append('/scratch')
from SparseNOvA import datasets

dataset = datasets.get_dataset(name='SparsePixelMapNOvA', filename='/data/mp5/cvnmap.parquet')

print(dataset[0])
print(dataset.data.caches)
print(dataset[100])
print(dataset.data.caches)

