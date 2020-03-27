#!/usr/bin/env python

import datasets

dataset = datasets.get_dataset(name='SparsePixelMapNOvA', filedir='/data/mp5')

print('Querying length... just returning a placeholder value for now:',len(dataset))

print('Querying first element... just returning a placeholder value for now:', dataset[0])

