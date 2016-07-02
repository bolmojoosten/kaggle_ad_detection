# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:17:53 2016

@author: joostbloom
"""
import pandas as pd
import numpy as np
import os
    
def load_train_data(forceReload=False, maxRows=20000, rs=123):
    return load_data('data/ItemPairs_train.csv', 'data/ItemInfo_train.csv',forceReload, maxRows, rs)

def load_test_data(forceReload=False, maxRows=20000, rs=123):
    return load_data('data/ItemPairs_test.csv', 'data/ItemInfo_test.csv',forceReload, maxRows, rs)


def load_data(pairdatafile, itemdatafile,forceReload=False, maxRows=20000, rs=123):

    #pairdatafile = 'data/ItemPairs_train.csv'
    pairdatafile_merged = samplefile_path(pairdatafile,'_merged')
    #itemdatafile = 'data/ItemInfo_train.csv'
    
    if os.path.isfile(pairdatafile_merged) and not forceReload:
        item_pairs = pd.read_csv(pairdatafile_merged)
        print("Loaded {} previously merged item pairs from {}".format(len(item_pairs),pairdatafile_merged))
        return (item_pairs, pairdatafile_merged)
         
    
    # Data Types
    itempairstypes = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
    }
    itemdatatypes = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }
    
    print("Loading item pairs")
    item_pairs = pd.read_csv(pairdatafile, dtype=itempairstypes)
    
    print("Loading itemdata")
    item_data = pd.read_csv(itemdatafile, dtype=itemdatatypes) 

    print("Merging item pairs and itemdata")
    item1_cols = [c+"_1" for c in item_data.columns]
    item2_cols = [c+"_2" for c in item_data.columns]
    
    #item1
    item1 = item_data.rename(columns = dict(zip(item_data.columns,item1_cols)))    
    item_pairs = pd.merge(item_pairs, item1, how='left',on='itemID_1', left_index=True)    
    del item1
    
    item2 = item_data.rename(columns = dict(zip(item_data.columns,item2_cols)))
    item_pairs = pd.merge(item_pairs, item2, how='left',on='itemID_2', left_index=True)
    del item2
    
    if maxRows>0:
        item_pairs = item_pairs.sample(maxRows, random_state=rs)
        item_pairs.to_csv(pairdatafile_merged)
    
    return (item_pairs, pairdatafile_merged)

def samplefile_path(filepath, extraString='_sample'):
    filename = filepath[ (filepath.rfind('/')+1) : filepath.rfind('.') ]
    path = filepath[0:(filepath.rfind('/')+1)]
    
    return path + filename + extraString + '.csv'
    

def make_small_subset(csvPath,maxRows = 20000, rs=123):
    
    savepath = samplefile_path(csvPath)
    
    tmpData = pd.read_csv(csvPath)
    
    tmpData.sample(maxRows, random_state=rs).to_csv(savepath)
    
    return savepath
    
def load_data_old(forceReload=False, maxRows=20000, rs=123):

    datafile = 'data/ItemPairs_train.csv'
    datafile_sample = samplefile_path(datafile,'merged')
    itemdatafile = 'data/ItemInfo_train.csv'
    itemdatafile_sample = samplefile_path(itemdatafile)
    
    # Data Types
    itempairstypes = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
    }
    itemdatatypes = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }
    
    if maxRows > 0:
        if not os.path.isfile(datafile_sample) or forceReload:
            print("Making subset of item pair data {}".format(maxRows))
            make_small_subset(datafile, maxRows, rs)
        
        item_pairs = pd.read_csv(datafile_sample, dtype=itempairstypes)
    else:
        item_pairs = pd.read_csv(datafile, dtype=itempairstypes)
    
    
    if maxRows > 0:
        prodIds = pd.concat([ item_pairs.itemID_1, item_pairs.itemID_2]).unique()
        
        if not os.path.isfile(itemdatafile_sample) or forceReload:
            print("Making subset of item data {}".format(len(prodIds)))
            #tmpData = pd.read_csv(itemdatafile, index_col='itemID', dtype=itemdatatypes)
            tmpData = pd.read_csv(itemdatafile, dtype=itemdatatypes)
            
            prods = tmpData.loc[prodIds]
            prods.to_csv(itemdatafile_sample)
        
            del tmpData
        
        #item_data = pd.read_csv(itemdatafile_sample,index_col='itemID', dtype=itemdatatypes)
        item_data = pd.read_csv(itemdatafile_sample, dtype=itemdatatypes)
    else:
        
        #item_data = pd.read_csv(itemdatafile,index_col='itemID', dtype=itemdatatypes)
        item_data = pd.read_csv(itemdatafile, dtype=itemdatatypes)  

    print("Merging items pairs and itemdata")
    item1_cols = [c+"_1" for c in item_data.columns]
    item2_cols = [c+"_2" for c in item_data.columns]
    
    #item1
    item1 = item_data.rename(columns = dict(zip(item_data.columns,item1_cols)))    
    item_pairs = pd.merge(item_pairs, item1, how='left',on='itemID_1', left_index=True)    
    del item1
    
    item2 = item_data.rename(columns = dict(zip(item_data.columns,item2_cols)))
    item_pairs = pd.merge(item_pairs, item2, how='left',on='itemID_2', left_index=True)
    del item2
    
    if maxRows>0:
        item_pairs = item_pairs.sample(maxRows, random_state=rs)
        item_pairs.to_csv()
    
    return (item_pairs, item_data)

def load_data_test():

    # Read related data
    #cat_data = pd.read_csv('data/Category.csv')
    #oc_data = pd.read_csv('data/Location.csv')
    #pairs_test = pd.read_csv('data/ItemPairs_test.csv')
    #pairs_train = pd.read_csv('data/ItemPairs_train.csv')

    datafile = 'data/ItemPairs_test.csv'
        
    item_pairs = pd.read_csv(datafile)
    
    itemdatafile = 'data/ItemInfo_test.csv'

    
    item_data = pd.read_csv(itemdatafile,index_col='itemID')
    
    return (item_pairs, item_data)