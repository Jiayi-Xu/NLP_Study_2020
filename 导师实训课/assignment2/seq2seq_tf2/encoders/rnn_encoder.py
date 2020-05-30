import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        # 用bigru
        self.enc_units = enc_units // 2
        """
        定义Embedding层，加载预训练的词向量
        your code
        """

        """        
        tf.keras.layers.Embedding(
            input_dim,  词汇表大小
            output_dim  词向量维度
            )
        Input shape:
            2D tensor with shape: (batch_size, input_length) -> x = [16,200]
        Embedding是一个层，继承自Layer，Layer有weights参数，weights参数是一个list，
            里面的元素都是numpy数组。在调用Layer的构造函数的时候，weights参数就被存储到了_initial_weights变量
        """

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        # tf.keras.layers.GRU自动匹配cpu、gpu
        """
        定义单向的RNN、GRU、LSTM层
        your code
        参数初始化glorot_uniform:从 [-limit，limit] 中的均匀分布中抽取样本
        """
        # 获取是否gpu配置
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

        if gpus:
            self.gru = tf.compat.v1.keras.layers.CuDNNGRU(self.enc_units, return_sequences=True, return_state=True,
                                                          recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                           recurrent_initializer='glorot_uniform')

        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    # call : 接受每一个单词word embedding 和 上一个时间点的hidden state。输出的是这个时间点的hidden state
    # ht = RNNenc(xt,ht-1)
    def call(self, x, hidden):
        # print("embedding之前x为", x, x.shape)
        # 输入格式为batch_size, input_length -> [16,200]
        # 输出格式为batch_size, input_length, output_dim -> [16,200,256]

        x = self.embedding(x)

        # print("embedding之后x为 (batch_size, input_length, output_dim)", x.shape)

        # hidden 为 (batch_size, output_dim) (16, 256)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)

        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)

        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
    
