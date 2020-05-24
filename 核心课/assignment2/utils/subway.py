import requests
from bs4 import BeautifulSoup
import re
import getGPS
import pickle

findLine = r'<div class="subway_num.*>(.*?)</div>'
findStation = r'<div class="station.*">(.*?)<.*'
# findAll = r'<div class="(?:station|subway_num).*">(.*?)<.*'
def main():
    # 爬北京地铁站获取地铁线路站点信息
    baseurl = 'https://www.bjsubway.com/station/xltcx/'
    # 调用getData函数 获取
    relDic,geoDic = getData(baseurl)
    savepath = "stations.pkl"
    saveData(relDic,savepath)
    savepath2 = "gps.pkl"
    saveData(geoDic,savepath2)

def getData(baseurl):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:63.0) Gecko/20100101 Firefox/63.0',
    }

    # Get方式获取网页数据
    html = requests.get(baseurl, headers=headers)
    html.encoding = 'GBK'
    bs = BeautifulSoup(html.text, "html.parser")

    item = bs.find_all("div", class_="line_content")
    item = str(item)
    # 获取地铁线路和站名
    line = re.findall(findLine, item)
    station = re.findall(findStation, item)
    # 存储线路和站点关系数据
    relDic = getMapDic(line, station)
    print(relDic)
    # 存储通过百度api返回的站点gps数据
    geoDic = getGeoDic(station)
    print(geoDic)
    return relDic,geoDic

def saveData(dataList, savepath):
    with open(savepath,'wb') as f:
        pickle.dump(dataList, f)

# 根据地铁站名获取站点gps坐标数据
def getGeoDic(station):
    geoDic = {}
    for s in station:
        # 调用百度api获取地铁站的gps坐标
        geoCoord = getGPS.find_location(s, '北京')
        geoDic[s] = geoCoord
    return geoDic
# 获取地铁线路和站点关系数据
def getMapDic(line,station):
    # 地铁站站数
    lst = [23, 18, 24, 23, 34, 29, 32, 13, 45, 17, 30, 20, 10, 14, 12, 14, 11, 4, 12, 7]
    dic = {}
    staLst = []
    prev = 0
    for i in lst:
        # 截取station列表里对应地铁线的站点
        staLst.append(station[prev:prev + i])
        prev += i
    for i, l in enumerate(line):
        dic[l] = staLst[i]
    return dic
# 入口函数
if __name__=="__main__":
    main()