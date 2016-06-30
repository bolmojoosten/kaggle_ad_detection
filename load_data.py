# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:17:53 2016

@author: joostbloom
"""
import pandas as pd
import os


def make_small_subset(csvPath,maxRows = 20000, rs=123):
    filename = csvPath[ (csvPath.rfind('/')+1) : csvPath.rfind('.') ]
    path = csvPath[0:(csvPath.rfind('/')+1)]
    
    savepath = path + filename + '_sample.csv'
    
    tmpData = pd.read_csv(csvPath)
    
    tmpData.sample(maxRows, random_state=rs).to_csv(savepath)
    
    return savepath

def load_data(forceReload=False, maxRows=20000, rs=123):

    # Read related data
    #cat_data = pd.read_csv('data/Category.csv')
    #oc_data = pd.read_csv('data/Location.csv')
    #pairs_test = pd.read_csv('data/ItemPairs_test.csv')
    #pairs_train = pd.read_csv('data/ItemPairs_train.csv')

    datafile = 'data/ItemPairs_train_sample.csv'
    
    if not os.path.isfile(datafile) or forceReload:
        print("Making subset of item pair data {}".format(maxRows))
        make_small_subset('data/ItemPairs_train.csv', maxRows)
        
    item_pairs = pd.read_csv(datafile)
    
    prodIds = pd.concat([ item_pairs.itemID_1, item_pairs.itemID_2]).unique()
    
    itemdatafile = 'data/ItemInfo_train_sample.csv'

    if not os.path.isfile(itemdatafile) or forceReload:
        print("Making subset of item data {}".format(len(prodIds)))
        tmpData = pd.read_csv('data/ItemInfo_train.csv', index_col='itemID')
        
        prods = tmpData.loc[prodIds]
        prods.to_csv(itemdatafile)
    
        del tmpData
    
    item_data = pd.read_csv(itemdatafile,index_col='itemID')
    
    return (item_pairs, item_data)