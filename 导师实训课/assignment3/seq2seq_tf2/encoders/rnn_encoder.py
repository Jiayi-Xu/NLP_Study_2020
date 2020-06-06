import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2

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
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # tf.keras.layers.GRU自动匹配cpu、gpu
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        # [batch_size,max_enc_length,embed_size]
        # print("x.shape is", x.shape)

        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
    
