
# 执行

运行记录在文件 “华为云代码运行记录.ipynb”


# 预处理

## /utils/preprocess.py文件

+ 对train和test文本进行解析数据

+ 对每一行调用preprocess\_sentence函数
    + 将句子用jieba进行分词
    + 移除remove_words里的词
    + 返回分词后的句子
    + 产出三个文件：
        + train\_set.seg\_x.txt，
        + train\_set.seg\_y.txt，
        + test\_set.seg\_x.txt

<!--（在设置停用词时候符号为中文符号：。？！等被移除）-->
+ 中文符号：。，？不删除，也是句子的组成。

## /utils/data_loader.py文件


1.读取preprocess产出的三个文件处理成词列表
lines = read_data('文件'))

2.建立词索引（默认按照词频排序）

+ 函数定义：build\_vocab(items, sort=True, min\_count=0, lower=False)
	+ sort默认为TRUE 按频率排序 
	+ min\_count排掉低频词
		+ 设置为3 vocab有45421条
		+ 设置为4 vocab有37465条
		+ 设置为5 vocab有32527条

vocab, reverse\_vocab = build\_vocab(lines, min\_count=5)

3.保存vocab.txt文件
save\_word\_dict(vocab, '{}/datasets/vocab.txt'.format(BASE\_DIR))


## /utils/build\_w2v.py文件

+ 函数包含：
	+ rread\_lines(path, col\_sep=' ')
	+ extract\_sentence(train\_x\_seg\_path, train\_y\_seg\_path, test\_seg\_path)
	+ save\_sentence(lines, sentence\_path)
	+ build(
		+ train\_x\_seg\_path, train\_y\_seg\_path, test\_seg\_path
		+ out\_path=None, #产出到word2vec.txt
		+ sentence\_path='', #产出到sentences.txt
       + w2v\_bin\_path="w2v.bin", #产出到utils目录下
       + min\_count=5)

## utils/data_utils.py

+ 函数load_word2vec被SequenceToSequence调用
	+ 读取word2vec.txt文件和vocab.txt文件
		+ word2vec\_dict 格式为 word : embedding 256维度
		+ vocab_dict 格式为 word : index
		+ embedding\_matrix = [vocab\_size,embed\_size] 这里维度对应30000*256 格式为index:embedding

# 序列模型 目录：/seq2seq\_tf2

## /bin/main.py
主程序入口

+ 定义了各种参数 变量名=params：
	+ 模型参数
	+ path 设置各种文件路径
	+ epoch和step参数
	+ mode参数：设置为train，model，greedy_decode
		+ 调用文件train\_eval\_test.py下的train(params) 

## train\_eval\_test.py

### train函数

+ 调用batcher.py文件下的类：Vocab(params["vocab\_path"], params["vocab\_size"])
	+ 读取文件vocab.txt 
	+ 建立word和id的索引关系
+ 调用batcher.py文件下的函数：batcher(vocab, hpm) 【一种为TensorFlow 模型创建输入管道的新方式。把数组、元组、张量等转换成DatasetV1Adapter格式】
	+ 调用batch_generator
		+ generator:为函数example_generator得到一个生成器
		+ vocab, 
		+ train\_x\_path, 
		+ train\_y\_path,
		+ test\_x\_path, 
		+ max\_enc\_len, 
		+ max\_dec\_len, 
		+ batch\_size, 
		+ mode)
+ 调用models.py下的类SequenceToSequence：model = SequenceToSequence(params)
	+ rnn\_encoder -> encoder
	+ rnn\_decoder -> [decoder, attention]
	+ data\_utils.py下的load_word2vec函数 -> embedding\_matrix \[vocab\_size*embed\_size]
	
### test函数

+ 建立词索引，引入model
+ predict_result(model, params, vocab, params['test_save_dir'])
	+ results = greedy_decode(model, dataset, vocab, params)
		+ batch\_greedy\_decode(model, enc_data, vocab, params)  



## rnn_encoder.py



## batcher.py

Vocab类：

+ id2word变量形式为 word : 索引
+ enc\_input = [vocab.word\_to\_id(w) for w in article_words] 句子的词的索引
+ dec\_input, target = get\_dec\_inp\_targ\_seqs(abs\_ids, max\_dec\_len, start\_decoding, stop_decoding)
	+ 例子：
		+ dec\_input is [2, 14, 196, 212, 38, 196, 93, 7, 4, 1912, 1913] 多了START\_DECODING(2)
		+ target is [14, 196, 212, 38, 196, 93, 7, 4, 1912, 1913, 3] 多了STOP_DECODING(3)


## 用到的函数

+ https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Embedding?hl=en
	+ embedding介绍

+ https://tensorflow.google.cn/api_docs/python/tf/split?hl=en
	+ tf.split(value, num\_or\_size\_splits, axis=0, num=None, name='split'
)