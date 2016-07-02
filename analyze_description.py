# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:11:41 2016

@author: guusjeschouten
"""
import time
from load_data import load_train_data
# Settings for loading data
maxRows = 20000 #<=0 to read all data
random_state = 99
reloadData = False

labels = ['Duplicates','Originals']

start = time.time()
(data, path) = load_train_data(forceReload=reloadData,maxRows=maxRows, rs=random_state)
print("Loading data ({} rows) took {:.2f}s".format(data.shape[0], time.time()-start))

data['description_words_1'] = data['description_1'].apply(lambda x: set(x.split(" ")))
data['description_words_2'] = data['description_2'].apply(lambda x: set(x.split(" ")))

# Calculate number of similar words
c1 = 'description_words_1'
c2 = 'description_words_2'
data['description_same_words'] = data[[c1,c2]].apply(lambda x: float(len( set(x[c1]) & set(x[c2]) )) / len(x[c1]), axis=1)

# Text similarity using NLP

def np_text_similar(x):

    string1 = x[0]
    string2 = x[1]    
    
    vect = TfidfVectorizer(min_df=1)    
    try:
        tfidf = vect.fit_transform([string1,string2])
        return (tfidf * tfidf.T).A[1][0]

    except:
        print(string1 + " vs " + string2 + " failed")
        return 0
    
    
# More advanced text analysis
from sklearn.feature_extraction.text import TfidfVectorizer
start = time.time()
data['description_similar'] = data[['description_1','description_2']].apply(np_text_similar, axis=1)
print("Text analysis took {:.2f}s".format(time.time()-start))



c = 'description_same_words'
d1 = data[data.isDuplicate==1]
d2 = data[data.isDuplicate==0]

print("Average word difference dup {} vs ori {}".format(np.mean(d1[c]),np.mean(d2[c])))

plt.figure()
d1[c1].apply(lambda x: len(x)).plot.hist(bins=30, alpha=0.5)
d2[c1].apply(lambda x: len(x)).plot.hist(bins=30, alpha=0.5)
plt.xlabel('Number of words')
plt.legend(labels)

plt.figure()
d1[c].plot.hist(bins=30, alpha=0.5)
d2[c].plot.hist(bins=30, alpha=0.5)
plt.xlabel('Number of same words')
plt.legend(labels)

plt.figure()
d1['description_similar'].plot.hist(bins=30, alpha=0.5)
d2['description_similar'].plot.hist(bins=30, alpha=0.5)
plt.xlabel('Text similarity')
plt.legend(labels)



    