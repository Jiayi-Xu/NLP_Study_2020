import pandas
from bs4 import BeautifulSoup 
from urllib import request
import re
import pandas as pd
import numpy as np
import urllib.parse as urp
from xml.etree import ElementTree

def find_location(name,city):
        my_ak = '3RifGxMcgC2BFO6BVOpjY7H6lGiPIl37'
        tag = urp.quote('地铁站')
        qurey = urp.quote(name)
        try:
            url = 'http://api.map.baidu.com/place/v2/search?query='+qurey+'&tag='+'&region='+urp.quote(city)+'&output=json&ak='+my_ak
            req = request.urlopen(url)
            res = req.read().decode()
            lat = pd.to_numeric(re.findall('"lat":(.*)',res)[0].split(',')[0])
            lng = pd.to_numeric(re.findall('"lng":(.*)',res)[0])
            return (lng,lat)  #经度和纬度
        except:
            return 0,0

if __name__ == "__main__":
    print(find_location('公主坟地铁站','北京'))