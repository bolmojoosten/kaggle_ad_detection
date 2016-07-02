# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:11:41 2016

@author: guusjeschouten
"""

from load_data import load_train_data
# Settings for loading data
maxRows = 20000 #<=0 to read all data
random_state = 99
reloadData = False

start = time.time()
(data, path) = load_train_data(forceReload=reloadData,maxRows=maxRows, rs=random_state)
print("Loading data ({} rows) took {:.2f}s".format(data.shape[0], time.time()-start))
