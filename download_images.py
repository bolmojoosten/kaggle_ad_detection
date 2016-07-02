# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:15:58 2016

@author: guusjeschouten
"""

import urllib, urllib2, cookielib

import requests
url = 'https://www.kaggle.com/account/login'
values = {'username': 'bolmovic',
          'password': 'Sm3efUrfA8Qy'}

r = requests.post(url, data=values)
r = requests.post(url, data=values)

def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    print r.content
    print(local_filename)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename
    
#download_file('https://www.kaggle.com/c/avito-duplicate-ads-detection/download/Images_0.zip')
    
cj = cookielib.CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
login_data = urllib.urlencode(values)
opener.open(url, login_data)
resp = opener.open('https://www.kaggle.com/c/avito-duplicate-ads-detection/download/Category.csv.zip')
print resp.read()