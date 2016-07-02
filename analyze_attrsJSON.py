# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:11:41 2016

@author: guusjeschouten
"""
import time
import json
from load_data import load_train_data
# Settings for loading data
maxRows = 20000 #<=0 to read all data
random_state = 99
reloadData = False

labels = ['Duplicates','Originals']

start = time.time()
(data, path) = load_train_data(forceReload=reloadData,maxRows=maxRows, rs=random_state)
print("Loading data ({} rows) took {:.2f}s".format(data.shape[0], time.time()-start))

for c in data.columns:
    if data[c].dtype=='int64':
        data[c].fillna(0,inplace=True)
    elif data[c].dtype=='float64':
        data[c].fillna(0,inplace=True)
    elif data[c].dtype=='object':
        data[c].fillna("",inplace=True)
        
def jsontodict(str_json):
    if len(str_json)==0:
        return {}
    else:
        return json.loads(str_json)

data.loc[:,'attrsJSON_1'] = data['attrsJSON_1'].apply(lambda x: jsontodict(x))
data.loc[:,'attrsJSON_2'] = data['attrsJSON_2'].apply(lambda x: jsontodict(x))



data1 = data[data.isDuplicate==1]
data2 = data[data.isDuplicate==0]

col_name = 'attrsJSON'
c1 = col_name + '_1'
c2 = col_name + '_2'

ld = len(data1)
lnd = len(data2)

d = float(sum(data1[c1]==data1[c2]))
nd = float(sum(data2[c1]==data2[c2]))

print("{}: d: {:.2f} vs. nd: {:.2f}".format(col_name,d/ld, nd/lnd))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
# Analyse occurance of keys
key_list = []
key_count = {}
key_count_d = {}
key_count_nd = {}
key_type = {}
key_same_d = {}
key_same_nd = {}
key_inboth = {}

for i,v in data.iterrows():
    keys = v[c1].keys()
    
    isDup = v['isDuplicate']
    
    key_list = list( set(key_list) | set(keys))
    
    for k in keys:
        key_type[k] = is_number(v[c1][k])
        keyinOther = k in v[c2]
        
        if k in key_count:
            key_count[k] += 1
            if isDup:
                key_count_d[k] += 1
            else:
                key_count_nd[k] += 1
            
            if keyinOther:
                key_inboth[k] +=1                
                
                if (v[c1][k] == v[c2][k]):
                    key_same_d[k] += 1 * isDup
                    key_same_nd[k] += 1 * (not isDup)
            
        else:
            key_count[k] = 1
            key_count_d[k] = isDup
            key_count_nd[k] = isDup
            key_inboth[k] = keyinOther
            if keyinOther:
                key_same_d[k] = (isDup & (v[c1][k] == v[c2][k])) * 1
                key_same_nd[k] = (not isDup & (v[c1][k] == v[c2][k])) * 1
            else:
                key_same_d[k] = isDup
                key_same_nd[k] = isDup
plt.figure()
plt.scatter(range(len(key_count_d)), key_count_d.values(),c='r')
plt.scatter(range(len(key_count_nd)), key_count_nd.values(),c='g') 
plt.ylabel('Key count')
plt.xlabel('Key')
plt.legend(labels) 

plt.figure()
plt.scatter(range(len(key_count_d)), key_same_d.values(),c='r')
plt.scatter(range(len(key_count_nd)), key_same_nd.values(),c='g')
plt.scatter(range(len(key_count_nd)), np.subtract(key_same_nd.values(),key_same_d.values()),c='b')  
plt.ylabel('Same value count')
plt.xlabel('Key')
#plt.legend(labels) 
plt.figure()
plt.scatter(range(len(key_count_d)), key_inboth.values(),c='r')
plt.ylabel('Keys present in both items')
plt.xlabel('Key')

plt.hist(np.subtract(key_same_nd.values(),key_same_d.values()), bins=100)

 
nKeys = data[c1].apply(lambda x: len(x.keys()))
nKeys1 = data1[c1].apply(lambda x: len(x.keys()))
nKeys2 = data2[c1].apply(lambda x: len(x.keys()))
nSameKeys1 = data1[[c1,c2]].apply(lambda x: len(set(x[0].keys()) & set(x[1].keys() )),axis=1)
nSameKeys2 = data2[[c1,c2]].apply(lambda x: len(set(x[0].keys()) & set(x[1].keys() )),axis=1)
nSameValues1 = data1[[c1,c2]].apply(lambda x: len(set(x[0].values()) & set(x[1].values() )),axis=1)
nSameValues2 = data2[[c1,c2]].apply(lambda x: len(set(x[0].values()) & set(x[1].values() )),axis=1)
plt.figure()
plt.hist(nKeys1, bins=50,alpha=0.5) 
plt.hist(nKeys2, bins=50, alpha=0.5) 
plt.grid()
plt.xlabel('Number of keys')
plt.legend(labels)
plt.ylabel('Count')

plt.figure()
plt.hist(nSameKeys1, bins=50,alpha=0.5) 
plt.hist(nSameKeys2, bins=50, alpha=0.5) 
plt.grid()
plt.xlabel('Number of same keys')
plt.legend(labels)
plt.ylabel('Count')

plt.figure()
plt.hist(np.subtract(nSameKeys1,nKeys1), bins=25,alpha=0.5) 
plt.hist(np.subtract(nSameKeys2,nKeys2), bins=25, alpha=0.5) 
plt.grid()
plt.xlabel('Relative Number of same keys')
plt.legend(labels)
plt.ylabel('Count')

plt.figure()
plt.hist(nSameValues1, bins=50,alpha=0.5) 
plt.hist(nSameValues2, bins=50, alpha=0.5) 
plt.grid()
plt.xlabel('Number of same values')
plt.legend(labels)
plt.ylabel('Count')

l = sorted(nKeys.unique())

l_k1 = [] # Average number of same keys for duplicates
l_k2 = [] # Average number of same keys for originals
l_v1 = [] # Average number of same values for duplicates
l_v2 = [] # Average number of same values for originals

for i in l:
    
    # Get ads with specific number of keys
    l_k1.append( np.mean(nSameKeys1[nKeys1==i]) )
    l_k2.append( np.mean(nSameKeys2[nKeys2==i]) )
    l_v1.append( np.mean(nSameValues1[nKeys1==i]) )
    l_v2.append( np.mean(nSameValues2[nKeys2==i]) )


plt.figure()
plt.plot(l,l_k1,'r')
plt.plot(l,l_v1,'r--')
plt.plot(l,l_k2,'g')
plt.plot(l,l_v2,'g--')
plt.grid()
plt.legend(['Keys dupl','Values dupl','Keys ori','Values ori'])
plt.xlabel('Number of keys')
plt.ylabel('Keys/values in common')

plt.figure()
plt.plot(l,np.divide(l_k1,l),'r')
plt.plot(l,np.divide(l_k2,l),'r--')
plt.plot(l,np.divide(l_v1,l),'g')
plt.plot(l,np.divide(l_v2,l),'g--')
plt.grid()
plt.legend(['Keys dupl','Values dupl','Keys ori','Values ori'])
plt.xlabel('Number of keys')
plt.ylabel('Relative number of keys/values in common')



    