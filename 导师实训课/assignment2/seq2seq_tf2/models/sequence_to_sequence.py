import tensorflow as tf
from seq2seq_tf2.encoders import rnn_encoder
from seq2seq_tf2.decoders import rnn_decoder
from utils.data_utils import load_word2vec
import time


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        # load pretrain word2vec weight matrix
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        """
        batch_size  : 256
        embed_size  : 256
        enc_units   : 256
        dec_units   : 256
        attn_units  : 256
        """
        self.encoder = rnn_encoder.Encoder(params["vocab_size"],
                                           params["embed_size"],
                                           params["enc_units"],
                                           params["batch_size"],
                                           self.embedding_matrix)

        # 计算score用的是BahdanauAttention 相加的计算方式
        self.attention = rnn_decoder.BahdanauAttention(params["attn_units"])

        self.decoder = rnn_decoder.Decoder(params["vocab_size"],
                                           params["embed_size"],
                                           params["dec_units"],
                                           params["batch_size"],
                                           self.embedding_matrix)
    # enc_inp 就是x [256，200】 输入256句每批次，序列长度为200
    # 第一步需要先计算出第一个enc_output
    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)

        # print("计算出第一个enc_output")
        # print("enc_output 维度为 (batch_size, input_length, hidden size) ", enc_output.shape)
        # print("enc_hidden 维度为 (batch_size, hidden size) ", enc_hidden.shape)

        return enc_output, enc_hidden
    
    def call(self, enc_output, dec_inp, dec_hidden, dec_tar):
        predictions = []
        attentions = []

        # print("dec_hidden (batch_size, hidden size) is", dec_hidden.shape)
        # print("enc_output 维度为 (batch_size, input_length, hidden size) ", enc_output.shape)

        # print("开始计算初始化上下文变量：context_vector")
        # context_vector：维度为(batch_size, hidden_size) (16, 256)
        context_vector, _ = self.attention(dec_hidden, enc_output)
        # print("初始化上下文变量完成， 维度为(batch_size, hidden_size)", context_vector.shape)

        # 进入循环 decoder是一个词一个词预测
        for t in range(dec_tar.shape[1]):
            # Teachering Forcing
            """
            应用decoder来一步一步预测生成词语概论分布
            your code
            如：xxx = self.decoder(), 采用Teachering Forcing方法
            """
            # pred 最终预测词的概率
            _, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:,t],1), dec_hidden, enc_output, context_vector)

            context_vector, attn_dist = self.attention(dec_hidden, enc_output)
            
            predictions.append(pred)
            attentions.append(attn_dist)
            #  tf.stack(predictions, 1) 维度：(batch_size, 序列长度, vocab_size)
            #  len(predictions) = 40 pred维度 = (16, 30000)

        return tf.stack(predictions, 1), dec_hidden