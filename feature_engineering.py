# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:45:46 2016

@author: joostbloom
"""
import pandas as pd
import numpy as np
import os
import math
import time
import json

def load_features(itemdata, traindata=True, forceReload=False):
    
    featurePath = 'data/ItemPairs_train_features.csv' if traindata else 'data/ItemPairs_test_features.csv'
    
    if not os.path.isfile(featurePath) or forceReload:
        print('Recalculating features')
        
        itempairs = make_features(itemdata)

        itempairs.to_csv(featurePath)
    
    
    data = pd.read_csv(featurePath)
    
    if traindata:
        y = data.isDuplicate
        X = data.drop('isDuplicate', axis=1)
        
        return (X,y)
    else:
        return data
    
def make_features(data):
    
    # Simple pre processing
    for c in data.columns:
        if data[c].dtype=='int64':
            data[c].fillna(0,inplace=True)
        elif data[c].dtype=='float64':
            data[c].fillna(0,inplace=True)
        elif data[c].dtype=='object':
            data[c].fillna("",inplace=True)
            
    
    # Remove a couple of extreme precies
    data.loc[data['price_1']>1e6,'price_1'] = 1e6
    data.loc[data['price_2']>1e6,'price_2'] = 1.1e6
    
    # Change image array to array
    data.loc[:,'images_array_1'] = data['images_array_1'].apply(lambda x: x.split(','))
    data.loc[:,'images_array_2'] = data['images_array_2'].apply(lambda x: x.split(','))
    
    # Convert JSON attr to dictionaries
    data.loc[:,'attrsJSON_1'] = data['attrsJSON_1'].apply(lambda x: jsontodict(x))
    data.loc[:,'attrsJSON_2'] = data['attrsJSON_2'].apply(lambda x: jsontodict(x))
    
    # Remember cols to drop later (only new features will be included)
    # isDuplicate is not contained in test data
    if 'isDuplicate' in data.columns:
        old_cols = data.columns.drop('isDuplicate')
    else:
        old_cols = data.columns
    
    # Price
    print("Processing price features...")
    data.loc[:,'price_dist'] = data[['price_1','price_2']].apply(np_relative_distance_num, axis=1)
    
    # Title
    print("Processing title features...")
    data['title_equal'] = np.equal(data['title_1'], data['title_2'])
    data['title_same_len'] = np.equal(data['title_1'].str.len(), data['title_2'].str.len())
    data['title_dist'] = data[['title_1','title_2']].apply(np_relative_distance_len, axis=1)
    
    # Description
    print("Processing description features...")
    data['description_equal'] = np.equal(data['description_1'], data['description_2'])
    data['description_same_len'] = np.equal(data['description_1'].str.len(), data['description_2'].str.len())
    data['description_dist'] = data[['description_1','description_2']].apply(np_relative_distance_len, axis=1)
    
    # LocationID
    # TO-DO: Identify locations with high probability of duplicates
    
    # images_array
    print("Processing images features...")
    data['images_len1'] = data['images_array_1'].apply(lambda x: len(x))
    data['images_len2'] = data['images_array_2'].apply(lambda x: len(x))
    data['images_same_len'] = np.equal(data['images_len1'],data['images_len2'])
    data['images_same'] = np.equal(data['images_array_1'],data['images_array_2'])
    
    # attrsJSON
    print("Processing attrsJSON features...")
    c1 = 'attrsJSON_1'
    c2 = 'attrsJSON_2'
    data['attrs_len1'] = data[c1].apply(lambda x: len(x))
    data['attrs_len2'] = data[c2].apply(lambda x: len(x))
    data['attrs_same_len'] = np.equal(data['attrs_len1'],data['attrs_len2'])
    data['attrs_n_keys_same'] = data[[c1,c2]].apply(lambda x: len(set(x[0].keys()) & set(x[1].keys() )),axis=1)
    data['attrs_n_values_same'] = data[[c1,c2]].apply(lambda x: len(set(x[0].values()) & set(x[1].values() )),axis=1)
    
    
    return data.drop(old_cols, axis=1)

def make_features_old(itempairs, itemdata):
    
    # General preprocessing
    itemdata.fillna(-1, inplace=True)
    itemdata.attrsJSON[itemdata.attrsJSON.isnull()]=None
    itemdata.images_array[itemdata.images_array.isnull()]=None
    itemdata.description[itemdata.description.isnull()]=None
    itemdata.title[itemdata.title.isnull()]=None
    
    # Make room for features
    itempairs['d_price'] = None
    itempairs['d_categoryID'] = None
    itempairs['d_title'] = None
    itempairs['d_len_title'] = None
    itempairs['d_description'] = None
    itempairs['d_len_description'] = None
    itempairs['d_locationID'] = None
    itempairs['d_attrsJSON'] = None
    itempairs['d_len_attrsJSON'] = None
    itempairs['d_metroID'] = None
    itempairs['d_images'] = None
    itempairs['d_len_images'] = None
    
    start = time.time()
    # Calculate values for features
    for i,d in itempairs.iterrows():
        item1= itemdata.loc[d.itemID_1]
        item2 = itemdata.loc[d.itemID_2]
        
        itempairs.set_value(i,'d_price', relative_distance(item1.price, item2.price))
        # Only duplicates within categories    
        #data.set_value(i,'d_categoryID', item1.categoryID == item2.categoryID)
        itempairs.set_value(i,'d_title', (item1.title == item2.title) + 0)
        itempairs.set_value(i,'d_len_title', relative_distance(lenNaN(item1.title),lenNaN(item2.title)))
        itempairs.set_value(i,'d_description', (item1.description == item2.description) + 0)
        itempairs.set_value(i,'d_len_description', relative_distance(lenNaN(item1.description), lenNaN(item2.description)))
        # JSON attributes
        if not (item1.attrsJSON is None or item2.attrsJSON is None):
            attrs1 = json.loads(item1.attrsJSON)
            attrs2 = json.loads(item2.attrsJSON)
            
            itempairs.set_value(i,'attr_keys1', len(attrs1))
            itempairs.set_value(i,'attr_keys2', len(attrs2))
            itempairs.set_value(i,'d_len_keys_attr', relative_distance(len(attrs1.keys()), len(attrs2.keys())))
            
        itempairs.set_value(i,'d_len_attrsJSON', relative_distance(lenNaN(item1.attrsJSON),lenNaN(item2.attrsJSON)))    
        itempairs.set_value(i,'d_locationID', (item1.locationID == item2.locationID) + 0) 
        itempairs.set_value(i,'d_metroID', (item1.metroID == item2.metroID) + 0)
        
        # Images
        img1 = len(item1.images_array.split(",")) if item1.images_array!=-1 else 0
        img2 = len(item2.images_array.split(",")) if item2.images_array!=-1 else 0
        
        itempairs.set_value(i,'d_len_images', relative_distance(img1,img2))
        itempairs.set_value(i,'d_images', img1 == img2 + 0)
        
        if i % 50000 == 0:
            print "Iteration {} of {} in {}s".format(i,len(itempairs),time.time()-start)


    # Remove extreme values
    #itempairs = itempairs[itempairs.d_price < 1e5]

    return itempairs

def simple_feature_analysis(itempairs):

    # Do some quick analysis on features
    data_dup = itempairs[itempairs.isDuplicate==1]
    data_nodup = itempairs[itempairs.isDuplicate==0]

    print "Price difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_price),np.mean(data_nodup.d_price))
    print "Title difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_title),np.mean(data_nodup.d_title))
    print "Title length difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_len_title),np.mean(data_nodup.d_len_title))
    print "Description difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_description),np.mean(data_nodup.d_description))
    print "Description length difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_len_description),np.mean(data_nodup.d_len_description))
    print "Location difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_locationID),np.mean(data_nodup.d_locationID))
    print "attrsJSON difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_attrsJSON),np.mean(data_nodup.d_attrsJSON))
    print "attrsJSON length difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_len_attrsJSON),np.mean(data_nodup.d_len_attrsJSON))
    print "metroID length difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_metroID),np.mean(data_nodup.d_metroID))
    print "d_images length difference: duplicates: {:.2f} vs {:.2f} no duplicates".format(np.mean(data_dup.d_images),np.mean(data_nodup.d_images))


    #data_nodup.d_price.plot.hist(bins=50,alpha=0.5)
    #data_dup.d_price.plot.hist(bins=50,alpha=0.5)
    #data.plot.scatter(x='d_price',y='d_len_title',c='isDuplicate')
    

def lenNaN(value):
    return 0 if value is -1 else len(value)
    
def relative_distance(value1, value2):
    
    if value1 + value2 == 0:
        return 0
    elif value1 - value2 == 0:
        return 0
    else: 
        return math.log(float(abs(value1 - value2)) / abs(value1 + value2))
        #return float(abs(value1 - value2)) / abs(value1 + value2)
        
def jsontodict(str_json):
    if len(str_json)==0:
        return {}
    else:
        return json.loads(str_json)
        
def np_relative_distance_len(vals):
    value1 = len(vals[0])
    value2 = len(vals[1])
    
    if value1 + value2 == 0:
        return 0
    elif value1 - value2 == 0:
        return 0
    else: 
        return math.log(float(abs(value1 - value2)) / abs(value1 + value2))
        #return float(abs(value1 - value2)) / abs(value1 + value2)
        
def np_relative_distance_num(vals):
    value1 = vals[0]
    value2 = vals[1]
    
    if value1 + value2 == 0:
        return 0
    elif value1 - value2 == 0:
        return 0
    else: 
        return math.log(float(abs(value1 - value2)) / abs(value1 + value2))
