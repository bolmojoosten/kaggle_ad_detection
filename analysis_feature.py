# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:33:31 2016

@author: joostbloom
"""

from load_data import load_data

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import json

datafile = 'data/ItemPairs_train.csv'
itemdatafile = 'data/ItemInfo_train.csv'

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
itempairs = pd.read_csv(datafile, dtype=itempairstypes)    
itemdata = pd.read_csv(itemdatafile, dtype=itemdatatypes)
item1_col = [c+"_1" for c in itemdata.columns]
item2_col = [c+"_2" for c in itemdata.columns]

#item1
item1 = itemdata.rename(columns = dict(zip(itemdata.columns,item1_col)))

itempairs2 = pd.merge(itempairs, item1, how='left',on='itemID_1', left_index=True)

del item1
item2 = itemdata.rename(columns = dict(zip(itemdata.columns,item2_col)))
itempairs2 = pd.merge(itempairs2, item2, how='left',on='itemID_2', left_index=True)
del item2
data = itempairs2.sample(50000, random_state=123)
data.to_csv('data/ItemInfo_train_merged.csv')

data = pd.read_csv('data/ItemInfo_train_merged.csv')

for c in data.columns:
    if data[c].dtype=='int64':
        data[c].fillna(0,inplace=True)
    elif data[c].dtype=='float64':
        data[c].fillna(0,inplace=True)
    elif data[c].dtype=='object':
        data[c].fillna("",inplace=True)

data.loc[data['price_1']>1e6,'price_1'] = 1e6
data.loc[data['price_2']>1e6,'price_2'] = 1.1e6

def jsontodict(str_json):
    if len(str_json)==0:
        return {}
    else:
        return json.loads(str_json)

# Change image array
data.loc[:,'images_array_1'] = data['images_array_1'].apply(lambda x: x.split(','))
data.loc[:,'images_array_2'] = data['images_array_2'].apply(lambda x: x.split(','))
data.loc[:,'attrsJSON_1'] = data['attrsJSON_1'].apply(lambda x: jsontodict(x))
data.loc[:,'attrsJSON_2'] = data['attrsJSON_2'].apply(lambda x: jsontodict(x))

# Change attrJSON



labels = ['Duplicates','Originals']

import math




# Compare keys of dictionary





col_name = 'images_array'

def compare_columns_array(data1,data2,col_name):
    c1 = col_name + '_1'
    c2 = col_name + '_2'
    
    d1Arr1 = data1[c1].apply(lambda x: (x.split(',')))
    d1Arr2 = data1[c2].apply(lambda x: (x.split(',')))
    d2Arr1 = data2[c1].apply(lambda x: (x.split(',')))
    d2Arr2 = data2[c2].apply(lambda x: (x.split(',')))
    d1Arr1_l = data1[c1].apply(lambda x: len(x.split(',')))
    d1Arr2_l = data1[c2].apply(lambda x: len(x.split(',')))
    d2Arr1_l = data2[c1].apply(lambda x: len(x.split(',')))
    d2Arr2_l = data2[c2].apply(lambda x: len(x.split(',')))
    
    r1 = float(sum((d1Arr1==d1Arr2)))/len(data1)
    r2 = float(sum((d2Arr1==d2Arr2)))/len(data2)
    r1_l = float(sum((d1Arr1_l==d1Arr2_l)))/len(data1)
    r2_l = float(sum((d2Arr1_l==d2Arr2_l)))/len(data2)
    
    print("{}: d: {:.2f} vs. nd: {:.2f}".format(col_name,r1, r2))
    print("{}: d: {:.2f} vs. nd: {:.2f}".format(col_name,r1_l, r2_l))
    
    plt.figure()
    plt.hist(d1Arr1_l,alpha=0.5, weights=np.zeros_like(d1Arr1_l) + 1. / len(d1Arr1_l))
    plt.hist(d2Arr1_l,alpha=0.5, weights=np.zeros_like(d2Arr1_l) + 1. / len(d2Arr1_l))
    plt.xlabel('Number of images')
    plt.ylabel('Relative count')
    plt.legend(labels)
    plt.grid()
    
    l = data1[c1].apply(lambda x: len((x.split(',')))).unique()
    #data.apply(lambda x: ( len(set(x[c1].split(',')) & set(x[c2].split(',')) )),axis=1).plot.hist()
    
    l_r = [] # Ratio of duplicates per number of images 
    l_r1 = [] # Ratio of equal images for duplicates per number of images 
    l_r2 = [] # Ratio of equal images for originals per number of images 
    l_ar1 = [] # Average number of equal images
    l_ar2 = [] # Average number of equal images
    for i in l:
        # Loop over all unique image lengths
    
        # Duplicates item1 and item2
        id11 = d1Arr1[d1Arr1_l==i]
        id12 = d1Arr2[d1Arr1_l==i]
        
        # Originals item1 and item2
        id21 = d2Arr1[d2Arr1_l==i]
        id22 = d2Arr2[d2Arr1_l==i]
        
        l_r.append( float(len(id11)) / (len(id11) + len(id21)))
        l_r1.append( float(sum((id11==id12)))/len(id11) )
        l_r2.append( float(sum((id21==id22)))/len(id21) )
        
        equal_images_duplicates = [len( set(id11.iloc[ii]) & set(id12.iloc[ii]) ) for (ii,v) in enumerate(id11)]
        equal_images_originals = [len( set(id21.iloc[ii]) & set(id22.iloc[ii]) ) for (ii,v) in enumerate(id21)]
        
        l_ar1.append(np.mean(equal_images_duplicates))  
        l_ar2.append(np.mean(equal_images_originals))   
        
        print("Length {}".format(i))
        print("{} duplicates".format(len(id11)))
        print("{} originals".format(len(id21)))
        print("{} duplicates".format(np.mean(equal_images_duplicates)))
        print("{} originals".format(np.mean(equal_images_originals)))

def compare_columns_ID(data1,data2,col_name):

    c1 = col_name + '_1'
    c2 = col_name + '_2'
    
    ld = len(data1)
    lnd = len(data2)
    
    d = float(sum(data1[c1]==data1[c2]))
    nd = float(sum(data2[c1]==data2[c2]))
    
    print("{}: d: {:.2f} vs. nd: {:.2f}".format(col_name,d/ld, nd/lnd))
    
    ld = [] # Relative number of non duplicates with same locationID
    lnd = [] # Relative number of duplicates with same locationID
    lr = [] # Relative number of duplicates for this locationID
    
    l = data1[c1].unique()
    
    for i in l:
        
        id1 = data1[data1[c1]==i] 
        id2 = data2[data2[c1]==i]
    
        # Number of duplicates vs non duplicates
        lr.append( float(len(id1)) / (len(id1)+len(id2)) )
        
        if len(id1)==0:
            ld.append(0)
        else:
            ld.append( float(sum(id1[c1]==id1[c2])) / len(id1) )
        if len(id2)==0:
            lnd.append(0)
        else:
            lnd.append( float(sum(id2[c1]==id2[c2])) / len(id2) )
    
    plt.figure()
    data1[c1].plot.hist(bins=50,alpha=0.5, weights=np.zeros_like(data1[c1]) + 1. / data1[c1].size)
    data2[c1].plot.hist(bins=50,alpha=0.5, weights=np.zeros_like(data2[c1]) + 1. / data2[c1].size)
    plt.xlabel(col_name)
    plt.ylabel('Count')
    plt.legend(['Duplicates','No duplicates'])
    plt.grid()
    
    plt.figure()
    #plt.hist(l,lr,c='b')
    plt.hist(ld,alpha=0.5, weights=np.zeros_like(ld) + 1. / len(ld))
    plt.hist(lnd,alpha=0.5, weights=np.zeros_like(lnd) + 1. / len(lnd))
    plt.xlabel('Relative number with same locationID')
    plt.ylabel('Location ID')
    plt.legend(['Duplicates','No duplicates'])
    plt.grid()
    
    plt.figure()
    plt.hist(np.subtract(ld,lnd))
    plt.grid()
    plt.ylabel('Count')
    plt.xlabel('Difference duplicates and not duplicates')
        #return float(abs(value1 - value2)) / abs(value1 + value2)

# Price
# Contains very large values
def compare_columns_ints(data1, data2, col_name):
    c1 = col_name + '_1'
    c2 = col_name + '_2'
    
    ld = len(data1)
    lnd = len(data2)

    d = float(sum(data1[c1]==data1[c2]))
    nd = float(sum(data2[c1]==data2[c2]))
    d_dl = data1[[c1,c2]].apply(pd_len_relative_distance_int, axis=1)
    nd_dl = data2[[c1,c2]].apply(pd_len_relative_distance_int, axis=1)
    print("{}: d: {:.2f} vs. nd: {:.2f}".format(col_name,d/ld, nd/lnd))
    print("Relative distance {}: d: {:.2f} vs. nd: {:.2f}".format(col_name,np.mean(d_dl), np.mean(nd_dl)))
    
    plt.figure()
    d_dl.plot.hist(bins=50,alpha=0.5)
    nd_dl.plot.hist(bins=50,alpha=0.5)
    plt.xlabel('Relative distance')
    plt.ylabel('Count')
    plt.legend(['Duplicates','No duplicates'])
    plt.grid()
    
    mi = min(data1[c1])
    ma = min([ max(data1[c1]),max(data2[c1])])
    
    n_d = []
    n_nd = []
    dl = []
    ndl = []
    dl_dl = []
    ndl_dl = []
    
    l = np.linspace(mi,ma,50)
    
    for i in l:
        td = data1[data1[c1]>=i]
        tnd = data2[data2[c1]>=i]
        
        n_d.append(len(td))
        n_nd.append(len(tnd))
        
        dl.append( float(sum( td[c1]==td[c2] )) / len(td))
        ndl.append( float(sum( tnd[c1]==tnd[c2])) / len(tnd))
        
        dl_dl.append( np.mean(td[[c1,c2]].apply(pd_len_relative_distance_int, axis=1)))
        ndl_dl.append( np.mean(tnd[[c1,c2]].apply(pd_len_relative_distance_int, axis=1)))
        
    plt.figure()
    plt.plot(l,dl,'b', linewidth=2.0)
    plt.plot(l,ndl,'b--', linewidth=2.0)
    plt.plot(l,dl_dl,'g', linewidth=2.0)
    plt.plot(l,ndl_dl,'g--', linewidth=2.0)
    plt.grid()
    plt.legend(['Equal string duplicates','Equal string no duplicates','Relative distance length string duplicates','Relative distance length string no duplicates'],fontsize=10)
    plt.xlabel('Value')
    plt.ylabel('Relative number of same string')
    plt.title(col_name)
    
    plt.figure()
    plt.plot(l, np.subtract(dl,ndl),'r-')
    plt.plot(l, np.subtract(ndl_dl,dl_dl),'g-')
    plt.grid()
    plt.xlabel('Value')
    plt.ylabel('Difference between classes')
    plt.legend(['Equal string','Difference'],fontsize=10)
    plt.title(col_name)
    
    plt.figure()
    plt.plot(l,n_d,'r', linewidth=2.0)
    plt.plot(l,n_nd,'g', linewidth=2.0)
    plt.grid()
    plt.xlabel('Min value')
    plt.ylabel('Item count')
    plt.legend(['Duplicates','No Duplicates'],fontsize=10)
    plt.title(col_name)
# Title / description
def compare_columns_strings(data1, data2, col_name):
    c1 = col_name + '_1'
    c2 = col_name + '_2'
    
    # Analyse features 
    ld = len(data1)
    lnd = len(data2)
    
    
    d = float(sum(data1[c1]==data1[c2]))
    nd = float(sum(data2[c1]==data2[c2]))
    d_l = float(sum( data1[c1].str.len()==data1[c2].str.len() ))
    nd_l = float(sum( data2[c1].str.len()==data2[c2].str.len()))
    d_dl = data1[[c1,c2]].apply(pd_len_relative_distance, axis=1)
    nd_dl = data2[[c1,c2]].apply(pd_len_relative_distance, axis=1)
    
    print("{}: d: {:.2f} vs. nd: {:.2f}".format(col_name,d/ld, nd/lnd))
    print("Len {}: d: {:.2f} vs. nd: {:.2f}".format(col_name,d_l/ld, nd_l/lnd))
    print("Relative distance {}: d: {:.2f} vs. nd: {:.2f}".format(col_name,np.mean(d_dl), np.mean(nd_dl)))
    
    plt.figure()
    plt.hist(d_dl,bins=50,alpha=0.5)
    plt.hist(nd_dl,bins=50,alpha=0.5)
    plt.xlabel('Relative distance')
    plt.ylabel('Count')
    plt.legend(['Duplicates','No duplicates'])
    plt.grid()
    
    mi = min(data1[c1].str.len())
    ma = min([ max(data1[c1].str.len()),max(data2[c1].str.len())])
    
    
    dl = []
    ndl = []
    dl_l = []
    ndl_l = []
    dl_dl = []
    ndl_dl = []
    
    l = range(mi,ma,ma/25)
    
    for i in l:
        td = data1[data1[c1].str.len()>i]
        tnd = data2[data2[c1].str.len()>i]
        
        dl.append( float(sum( td[c1]==td[c2] )) / len(td))
        ndl.append( float(sum( tnd[c1]==tnd[c2])) / len(tnd))
        dl_l.append( float(sum( td[c1].str.len()==td[c2].str.len() )) / len(td))
        ndl_l.append( float(sum( tnd[c1].str.len()==tnd[c2].str.len())) / len(tnd) )
        
        dl_dl.append( np.mean(td[[c1,c2]].apply(pd_len_relative_distance, axis=1))  )
        ndl_dl.append( np.mean(tnd[[c1,c2]].apply(pd_len_relative_distance, axis=1))  )
        
    plt.figure()
    plt.plot(l,dl,'b', linewidth=2.0)
    plt.plot(l,ndl,'b--', linewidth=2.0)
    plt.plot(l,dl_l,'r', linewidth=2.0)
    plt.plot(l,ndl_l,'r--', linewidth=2.0)
    plt.plot(l,dl_dl,'g', linewidth=2.0)
    plt.plot(l,ndl_dl,'g--', linewidth=2.0)
    plt.grid()
    plt.legend(['Equal string duplicates','Equal string no duplicates','Equal length string duplicates','Equal length string no duplicates','Relative distance length string duplicates','Relative distance length string no duplicates'],fontsize=10)
    plt.xlabel('Length of string')
    plt.ylabel('Relative number of same string')
    plt.title(col_name)
    
    plt.figure()
    plt.plot(l, np.subtract(dl,ndl),'b-')
    plt.plot(l, np.subtract(dl_l,ndl_l),'r-')
    plt.plot(l, np.subtract(ndl_dl,dl_dl),'g-')
    plt.grid()
    plt.xlabel('Length of string')
    plt.ylabel('Difference between classes')
    plt.legend(['Equal string','Equal string length','Difference'],fontsize=10)
    plt.title(col_name)
    
def pd_len_relative_distance(vals):
    value1 = len(vals[0])
    value2 = len(vals[1])
    
    if value1 + value2 == 0:
        return 0
    elif value1 - value2 == 0:
        return 0
    else: 
        return math.log(float(abs(value1 - value2)) / abs(value1 + value2))
        #return float(abs(value1 - value2)) / abs(value1 + value2)
def pd_len_relative_distance_int(vals):
    value1 = vals[0]
    value2 = vals[1]
    
    if value1 + value2 == 0:
        return 0
    elif value1 - value2 == 0:
        return 0
    else: 
        return math.log(float(abs(value1 - value2)) / abs(value1 + value2))
