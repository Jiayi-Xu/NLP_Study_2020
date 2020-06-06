import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# filenames_in = '{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR) # 输入文件的文件地址
# filenames_out = '{}/datasets/AutoMaster_TrainSet_small.csv'.format(BASE_DIR) # 新文件的地址

# train = pd.read_csv(filenames_in, encoding='utf-8')
# rows = train.iloc[:50]

# rows.to_csv(filenames_out, encoding='utf-8') # 将数据写入新的csv文件


filenames_in2 = '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR) # 输入文件的文件地址
filenames_out2 = '{}/datasets/AutoMaster_TestSet_10.csv'.format(BASE_DIR) # 新文件的地址

test = pd.read_csv(filenames_in2, encoding='utf-8')
rows2 = test.iloc[:10]

rows2.to_csv(filenames_out2, encoding='utf-8') # 将数据写入新的csv文件