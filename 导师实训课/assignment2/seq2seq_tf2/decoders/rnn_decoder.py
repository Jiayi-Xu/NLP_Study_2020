import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # units这里为256
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output):
        """
        :param dec_hidden: shape=(16, 256)
        :param enc_output: shape=(16, 200, 256)
        :param enc_padding_mask: shape=(16, 200)
        :param use_coverage:
        :param prev_coverage: None
        :return:
        """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)  # shape=(16, 1, 256)
        # att_features = self.W1(enc_output) + self.W2(hidden_with_time_axis)

        # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
        """
        定义score 
        计算公式：Va转置* tanh(w1*ht+w2*hs)
        计算流程：ht−1 → at → ct → ht，它使用前一个位置t-1的state计算t时刻的ht
        BahdanauAttention对Encoder和Decoder的双向的RNN的state拼接起来作为输出
        your code
        """
        # 通过decoder的hidden states加上encoder的hidden states来计算一个分数，用于计算权重
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        # Calculate attention distribution
        """
        归一化score，得到attn_dist 为(16,200,1)
        your code
        """
        # 每一个encoder的hidden states对应的权重
        attn_dist = tf.nn.softmax(score, axis = 1) #(16, 200, 1)

        # context_vector shape after sum == (batch_size, hidden_size)

        # 上下文变量context vector是一个对于encoder输出的hidden states的一个加权平均。
        context_vector = attn_dist * enc_output  # shape=(16, 200, 256)
        # tf.reduce_sum 用于计算张量tensor沿着某一维度的和，可以在求和后降维。
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape=(16, 256)
        return context_vector, tf.squeeze(attn_dist, -1)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)

        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        # 获取是否gpu配置
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

        if gpus:
            self.gru = tf.compat.v1.keras.layers.CuDNNGRU(self.dec_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True,
                                           recurrent_initializer='glorot_uniform')

        # self.dropout = tf.keras.layers.Dropout(0.5)
        """
        定义最后的fc层，用于预测词的概率
        your code
        """
        # self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activation.softmax)
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.nn.softmax)

    # 接受目标句子里单词的word embedding，和上一个时间点的hidden state
    def call(self, x, hidden, enc_output, context_vector):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # print('x is ', x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output = self.dropout(output)
        out = self.fc(output)

        return x, out, state

