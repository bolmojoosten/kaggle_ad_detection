# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:21:07 2016

@author: joostbloom
"""

def create_submit_file_kaggle_avito(y_prob):
    subfile='data/submission.csv'
    
    f = open(subfile,'w')
    f.write('id,probability\n')
    for i in range(len(y_prob)):
        f.write('{},{}\n'.format(i,y_prob[i,1]))
    f.close()