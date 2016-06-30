# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:45:46 2016

@author: joostbloom
"""
import pandas as pd
import numpy as np
import os
import math

def load_features(itempairs, itemdata, forceReload=False):
    featurePath = 'data/ItemPairs_train_features.csv'
    
    if not os.path.isfile(featurePath) or forceReload:
        print('Recalculating features')
        
        itempairs = make_features(itempairs, itemdata)

        itempairs.to_csv(featurePath)
    
    return pd.read_csv(featurePath)
    

def make_features(itempairs, itemdata):
    
    # General preprocessing
    itemdata.attrsJSON[itemdata.attrsJSON.isnull()]=None
    itemdata.images_array[itemdata.images_array.isnull()]=None
    itemdata.description[itemdata.description.isnull()]=None
    
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
    
    # Calculate values for features
    for i,d in itempairs.iterrows():
        item1= itemdata.loc[d.itemID_1]
        item2 = itemdata.loc[d.itemID_2]
        
        itempairs.set_value(i,'d_price', relative_distance(item1.price, item2.price))
        # Only duplicates within categories    
        #data.set_value(i,'d_categoryID', item1.categoryID == item2.categoryID)
        itempairs.set_value(i,'d_title', item1.title == item2.title)
        itempairs.set_value(i,'d_len_title', relative_distance(len(item1.title),len(item2.title)))
        itempairs.set_value(i,'d_description', item1.description == item2.description)
        itempairs.set_value(i,'d_len_description', relative_distance(lenNaN(item1.description), lenNaN(item2.description)))
        itempairs.set_value(i,'d_attrsJSON', item1.attrsJSON == item2.attrsJSON)
        itempairs.set_value(i,'d_len_attrsJSON', relative_distance(lenNaN(item1.attrsJSON),lenNaN(item2.attrsJSON)))    
        itempairs.set_value(i,'d_locationID', item1.locationID == item2.locationID) 
        itempairs.set_value(i,'d_metroID', item1.metroID == item2.metroID)
        
        # Images
        img1 = len(item1.images_array.split(",")) if item1.images_array!=None else 0
        img2 = len(item2.images_array.split(",")) if item2.images_array!=None else 0
        
        itempairs.set_value(i,'d_len_images', relative_distance(img1,img2))
        itempairs.set_value(i,'d_images', img1 == img2)


    # Remove extreme values
    itempairs = itempairs[itempairs.d_price < 1e5]

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
    return 0 if value is None else len(value)
    
def relative_distance(value1, value2):
    
    if value1 + value2 == 0:
        return 0
    elif value1 - value2 == 0:
        return 0
    else: 
        return math.log(float(abs(value1 - value2)) / abs(value1 + value2))
        #return float(abs(value1 - value2)) / abs(value1 + value2)
