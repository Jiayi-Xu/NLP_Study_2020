from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from data_utils import dump_pkl
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_lines(path, col_sep=' '):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            # strip 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            line = line.strip()
            # print(line)
            '''
            暂时注释 对句子按空格进行分割，传入sentences
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            '''
            # col_sep非空，有设置分隔符
            if col_sep:
                if col_sep in line:
                    lines.append(line.split(col_sep))
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    # ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    # for line in lines:
    #     ret.append(line)
    # return ret
    return lines

def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            # f.write('%s\n' % line.strip())
            f.write('%s\n' % line)
    print('save sentence:%s' % sentence_path)

# 词向量的训练
def build(train_x_seg_path, test_y_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=5):
    # 读取三个文件源然后合并三个文件中的句子
    # 根据col_sep进行拆分词
    sentences = extract_sentence(train_x_seg_path, test_y_seg_path, test_seg_path)
    print(sentences[:5])
    save_sentence(sentences, sentence_path)

    print('train w2v model...')
    # train model
    """
    通过gensim工具完成word2vec的训练，输入格式采用sentences，使用skip-gram，embedding维度256
    your code
    w2v = （one line）
    """
    # 如果模型还未训练过，则开始训练，否则的话跳过训练，直接加载模型
    if not os.path.exists(w2v_bin_path):
        model = Word2Vec(sentences, size=256, window=3, min_count=min_count, workers=4)
        model.wv.save_word2vec_format(w2v_bin_path, binary=True)
        print("save %s ok." % w2v_bin_path)

    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)

    # test
    print(model['技师'])
    sim = model.wv.similarity(u'技师', u'车主')
    print('技师 vs 车主 similarity score:', sim)
    # 打印出来为0.764

    # 存储词向量数据
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    #     为什么后缀不是pkl
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
          '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
          '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR))

