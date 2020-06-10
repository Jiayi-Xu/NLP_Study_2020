import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from tokenizer import segment
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 移除无用词常量 检查原始数据【语音】【图片】不包含信息，可以移除
REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']

# 定义读取停用词的函数
# 需要注意的是 停用词为中文符号
def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines

# 定义移除无用词的函数 jieba.lcut 返回的是list
def remove_words(words_list):
    # print("处理之前：",words_list)
    # 原始 words_list = [word for word in words_list if word not in REMOVE_WORDS]
    words_list = [word for word in words_list if word.strip() not in REMOVE_WORDS]
    # print("处理之后：",words_list)
    return words_list

"""
对train和test文本进行解析数据

对每一行调用preprocess_sentence函数
    将句子用jieba进行分词
    移除remove_words里的词
    返回分词后的句子
    产出三个文件：
        train_set.seg_x.txt，
        train_set.seg_y.txt，
        test_set.seg_x.txt
"""

def parse_data(train_path, test_path ):
    # 处理训练数据 X: Question和Dialogue Y: Report
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    train_df.fillna('', inplace=True)
    # 连接Question和Dialogue
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    print('train_x is ', len(train_x))
    # 对训练数据的每一行进行预处理
    train_x = train_x.apply(preprocess_sentence)
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    print('train_y is ', len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))
    # if 'Report' in train_df.columns:
        # train_y = train_df.Report
        # print('train_y is ', len(train_y))

    # 处理测试数据
    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    # 将列Dialogue和Question合并 作为数据X
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    test_y = []
    train_x.to_csv('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)

# 句子预处理：jieba分词 移除无用词 并用空格连接
def preprocess_sentence(sentence):
    # segment(sentence, cut_type='word', pos=False) 使用了jieba.lcut(sentence)
    # jieba.lcut 直接生成的就是一个list
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    # stopwords = read_stopwords('{}/datasets/stopwords.txt'.format(BASE_DIR))
    # REMOVE_WORDS.extend(stopwords)


    # 需要更换成自己数据的存储地址
    # 对数据进行预处理，再调用preprocess_sentence处理句子

    parse_data('{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR))

    # 小文件用于测试跑通程序用
    # parse_data('{}/datasets/AutoMaster_TrainSet_small.csv'.format(BASE_DIR),
    #            '{}/datasets/AutoMaster_TestSet_small.csv'.format(BASE_DIR))


