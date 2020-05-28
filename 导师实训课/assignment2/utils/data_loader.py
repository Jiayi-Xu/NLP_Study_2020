from collections import defaultdict
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            # update by xjy (words = line.split()) 默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
            words += line.split(' ')

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        """
        按照字典里的词频进行排序，出现次数多的排在前面
        your code(one line)
        """
        # reverse= True 按value从大到小排序
        dic = sorted(dic.items(), key=lambda x:x[1], reverse= True)
        # 对dic进行枚举遍历
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    """
    建立项目的vocab和reverse_vocab，vocab的结构是（词，index）
    your code
    vocab = (one line)
    reverse_vocab = (one line)
    """
    vocab = [ (item, index) for index, item in enumerate(result)]
    reverse_vocab = [(index, item) for index, item in enumerate(result)]

    return vocab, reverse_vocab


if __name__ == '__main__':
    """
    这三个文件由程序preprocess产出，格式为jieba分词后再移除停用词 ' '.join成的句子
        train_set.seg_x.txt，
        train_set.seg_y.txt，
        test_set.seg_x.txt
    调用read_data函数返回的是包含train_x,train_y,test_x的词列表
    """
    lines = read_data('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
                      '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
                      '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR))
    # sort默认为TRUE 按频率排序
    vocab, reverse_vocab = build_vocab(lines, min_count=5)
    # 为什么不存储reverse_vocab？ 如果要存储，需要更新函数save_word_dict里前后顺序
    save_word_dict(vocab, '{}/datasets/vocab.txt'.format(BASE_DIR))
