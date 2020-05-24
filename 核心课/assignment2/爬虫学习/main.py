import requests
from bs4 import BeautifulSoup
import re

findLink = re.compile(r'<a href="(.*?)">')
findImgSrc = re.compile(r'<img.*src="(.*?)"',re.S) # re.S让换行符包含在字符中
findTitle = re.compile(r'<span class="title">(.*)</span>')
findRank = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
findDes = re.compile(r'<span class="inq">(.*?)</span>')
def main():

    baseurl = 'https://movie.douban.com/top250?start='
    dataList = getData(baseurl)
    savepath = "douban.txt"
    saveData(dataList,savepath)

def getData(baseurl):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:63.0) Gecko/20100101 Firefox/63.0',
    }

    dataList = []
    for i in range(10):
        url = baseurl + str(i*25)
        print(url)
        # Get方式获取网页数据
        html = requests.get(url, headers=headers)
        bs = BeautifulSoup(html.text, "html.parser")

        for item in bs.find_all("div", class_="item"):
            item = str(item)
            # 获取影片详情链接
            link = re.findall(findLink, item)[0]
            # 获取图片源
            imgSrc = re.findall(findImgSrc, item)[0]
            # 获取影片名字
            title = re.findall(findTitle, item)[0]
            # 获取评分
            rank = re.findall(findRank, item)[0]
            # 获取概述
            if re.findall(findDes, item):
                des = re.findall(findDes, item)[0]
            else:
                des = ''
            line = [link,imgSrc,title,rank,des]
            print(link, imgSrc, title, rank, des)
            dataList.append(' '.join(line))
    return dataList

def saveData(dataList, savepath):
    with open(savepath,'w') as f:
        for line in dataList:
            f.write(line)

# 入口函数
if __name__=="__main__":
    main()