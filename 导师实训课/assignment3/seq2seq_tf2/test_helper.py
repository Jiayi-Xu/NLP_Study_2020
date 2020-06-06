import tensorflow as tf
import numpy as np
from seq2seq_tf2.batcher import output_to_words
from tqdm import tqdm
import math

# train_eval_test.predict_result调用greedy_decode
# def greedy_decode(model, dataset, vocab, params):
#     # 存储结果
#     batch_size = params["batch_size"]
#     results = []
#
#     # 测试集的数量
#     sample_size = params['sample_size']
#     # batch 操作轮数 math.ceil向上取整 小数 +1
#     # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
#     steps_epoch = sample_size // batch_size + 1
#
#     for i in tqdm(range(steps_epoch)):
#         # 只有x数据 没有标签y数据
#         enc_data, _ = next(iter(dataset))
#         # dataset[i * batch_size:(i + 1) * batch_size]
#         results += batch_greedy_decode(model, enc_data, vocab, params)
#     return results


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    batch_data = enc_data["enc_input"]
    batch_size = enc_data["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    # 输入只有X
    inputs = tf.convert_to_tensor(batch_data)

    # 构建encoder
    enc_output, enc_hidden = model.call_encoder(inputs)
    dec_hidden = enc_hidden
    # dec_input = tf.expand_dims([vocab.word_to_id(vocab.START_DECODING)] * batch_size, 1)

    # 第一步的dec_input为START_DECODING
    # 索引2 对应 START_DECODING
    dec_input = tf.constant([2] * batch_size)
    dec_input = tf.expand_dims(dec_input, axis=1)
    # print("第一步的dec_input是", dec_input)

    context_vector, _ = model.attention(dec_hidden, enc_output)
    # print("dec_hidden ",dec_hidden.shape)
    # print("enc_output ",enc_output.shape)
    for t in range(params['max_dec_len']):
        # 单步预测
        """
        your code, 通过调用decoder得到预测的概率分布
        """
        # print("ec_input[:, t] is",dec_input[:, t])
        # print("tf.expand_dims(dec_input[:, t], 1)",tf.expand_dims(dec_input[:, t], 1))
        _, pred, dec_hidden = model.decoder(dec_input,
                                           dec_hidden,
                                           enc_output,
                                           context_vector)
        # print("pred is",pred)
        context_vector, _ = model.attention(dec_hidden, enc_output)

        """
        your code, 通过调用tf.argmax完成greedy search，得到predicted_ids
        """
        # index for largest probability each row
        # 返回每行概率最大的索引值组成的列表
        predicted_ids = [tf.argmax(p).numpy() for p in pred]
        # print("predicted_ids is", predicted_ids)
        # predicted_ids is [1461, 1940, 1940, 1940, 1461, 1461, 1461, 1940, 1940, 1461]

        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '
        # print("predicts is",predicts)
        # predicts is ['舒畅 ', '运行 ', '运行 ', '运行 ', '舒畅 ', '舒畅 ', '舒畅 ', '运行 ', '运行 ', '舒畅 ']

        # using teacher forcing
        dec_input = tf.expand_dims(predicted_ids, 1)
        # print("下一步dec_input是",dec_input)

    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断vocab.word_to_id('[STOP]')
        if '[STOP]' in predict:
            # 截断stop
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    return results


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """
    # 修改参数，删除暂时不需要用到的参数
    def __init__(self, tokens, log_probs, state, attn_dists):
        self.tokens = tokens            # list of all the tokens from time 0 to the current time step t
        # 预测部分生成的概率值
        self.log_probs = log_probs      # list of the log probabilities of the tokens of the tokens
        # 从decoder传过来的状态变量
        self.state = state              # decoder state after the last token decoding
        self.attn_dists = attn_dists    # attention dists of all the tokens
        # generation probability of all the tokens

        self.abstract = ""
        # mark:后面算法需要用到的参数
        # self.p_gens = p_gens
        # self.coverage = coverage

        # self.text = ""
        # self.real_abstract = ""

    def extend(self, token, log_prob, state, attn_dist):
        """Method to extend the current hypothesis by adding the next decoded token and all
        the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          state=state,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist])
                          # we  add the attention dist of the decoded token

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)



def beam_decode(model, batch, vocab, params):
    # 删除暂时不需要用到的参数
    def decode_onestep(enc_outputs, dec_input, dec_state):
        """
            Method to decode the output step by step (used for beamSearch decoding)
            Args:
                sess : tf.Session object
                batch : current batch, shape = [beam_size, 1, vocab_size( + max_oov_len if pointer_gen)]
                (for the beam search decoding, batch_size = beam_size)
                enc_outputs : hiddens outputs computed by the encoder LSTM
                dec_state : beam_size-many list of decoder previous state, LSTMStateTuple objects,
                shape = [beam_size, 2, hidden_size]
                dec_input : decoder_input, the previous decoded batch_size-many words, shape = [beam_size, embed_size]
                cov_vec : beam_size-many list of previous coverage vector
            Returns: A dictionary of the results of all the ops computations (see below for more details)
        """
        # final_dists, dec_hidden, attentions = model(enc_outputs,  # shape=(3, 115, 256)
        #                                                                dec_state,  # shape=(3, 256)
        #                                                                enc_inp,  # shape=(3, 115)
        #                                                                enc_extended_inp,  # shape=(3, 115)
        #                                                                dec_input,  # shape=(3, 1)
        #                                                                batch_oov_len,  # shape=()
        #                                                                enc_pad_mask,  # shape=(3, 115)
        #                                                                use_coverage,
        #                                                                prev_coverage)  # shape=(3, 115, 1)
        # print("enc_outputs is",enc_outputs.shape)
        # print("dec_state is",dec_state.shape)
        context_vector, attention_weights = model.attention(dec_state, enc_outputs)
        _, final_dists, dec_hidden = model.decoder(dec_input,
                                           dec_state,
                                           enc_outputs,
                                           context_vector)
        # print("pred is",pred)

        # context_vector, attention_weights = model.attention(dec_hidden, enc_outputs)

        # tf.nn.top_k(input, k, name=None)
        # 返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
        # 设置params["beam_size"] * 2 为了并行计算 9*vocab-> 3 * (vocab * 3)
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_dists), k=params["beam_size"] * 2)
        # 计算log概率
        top_k_log_probs = tf.math.log(top_k_probs)

        results = {"dec_state": dec_hidden,
                   "attention_vec": attention_weights,  # [batch_sz, max_len_x, 1]
                   "top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs
                   }
        # 返回需要保存的中间结果和概率
        return results

    # end of the nested class

    # We run the encoder once and then we use the results to decode each time step token
    # state shape=(3, 256), enc_outputs shape=(3, 115, 256)

    # 计算第一个encoder的输出
    # print("batch enc_input is", batch["enc_input"])
    # enc_outputs, enc_hidden = model.call_encoder(batch["enc_input"])
    inputs = tf.convert_to_tensor(batch)
    enc_outputs, enc_hidden = model.call_encoder(inputs)
    dec_hidden = enc_hidden
    # print("刚开始 enc_outputs 为", enc_outputs.shape) # (3,115,256)
    # print("刚开始dec_hidden为",dec_hidden.shape)
    # print("dec_hidden[0] is",dec_hidden[0])

    # 初始化batch size个 假设对象 按 __init__里的传参顺序
    hyps = [Hypothesis(tokens=[vocab.word_to_id('[START]')],
                       log_probs=[0.0],
                       state=dec_hidden[0],
                       attn_dists=[]) for _ in range(params['batch_size'])]
    # print('hyps', len(hyps))
    # 初始化结果集
    results = []  # list to hold the top beam_size hypothesises
    steps = 0  # initial step

    while steps < params['max_dec_steps'] and len(results) < params['beam_size']:
        latest_tokens = [h.latest_token for h in hyps]  # latest token for each hypothesis , shape : [beam_size]
        # print('latest_tokens is ', latest_tokens)
        # we replace all the oov is by the unknown token
        # print(latest_tokens)
        latest_tokens = [t if t in range(params['vocab_size']) else vocab.word_to_id('[UNK]') for t in latest_tokens]
        # latest_tokens = [t if t in vocab.id2word else vocab.word2id('[UNK]') for t in latest_tokens]

        # we collect the last states for each hypothesis
        # print(latest_tokens)
        #
        # 获取所有隐藏层状态
        states = [h.state for h in hyps]
        # print('states i s', states)

        # we decode the top likely 2 x beam_size tokens tokens at time step t for each hypothesis
        # model, batch, vocab, params
        dec_input = tf.expand_dims(latest_tokens, axis=1)  # shape=(3, 1)
        # print('dec_input is ', dec_input)
        # print('step is ', steps)
        # print('dec_input is ', dec_input)
        # print('states is ', states)
        dec_states = tf.stack(states, axis=0)
        # print('dec_states is ', dec_states)
        # print('batch[0][enc_input] is ', batch[0]['enc_input'])
        # print('enc_outputs is ', enc_outputs)
        # print('dec_input is ', dec_input)
        # print('dec_states is ', dec_states)
        # print('batch[0][extended_enc_input is ', batch[0]['extended_enc_input'])  # problem maybe
        # print('batch[0][max_oov_len] is ', batch[0]['max_oov_len'])
        # print('batch[0][sample_encoder_pad_mask is ', batch[0]['sample_encoder_pad_mask'])
        # print('prev_coverage is ', prev_coverage)
        # print("dec_states shape",dec_states)
        # 单步运行decoder 计算需要的值
        decoder_results = decode_onestep(
            # batch[0]['enc_input'],  # shape=(3, 115)
                                 enc_outputs,  # shape=(3, 115, 256)
                                 dec_input,  # shape=(3, 1)
                                 dec_states)  # shape=(3, 115, 1)
        # print('returns["p_gen"] is ', returns["p_gen"])
        # print(np.squeeze(returns["p_gen"]))
        # np.squeeze(returns["p_gen"])
        # print('returns is ', returns["p_gen"])
        topk_ids = decoder_results['top_k_ids']
        topk_log_probs = decoder_results['top_k_log_probs']
        new_states = decoder_results['dec_state']
        attn_dists = decoder_results['attention_vec']


        # print('topk_ids is ', topk_ids)
        # print('topk_log_probs is ', topk_log_probs)

        # 现阶段全部可能情况
        all_hyps = []
        # 原有的可能情况数量
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        num = 1
        # print('num_orig_hyps is ', num_orig_hyps)
        for i in range(num_orig_hyps):
            # h, new_state, attn_dist, p_gen, coverage = hyps[i], new_states[i], attn_dists[i], p_gens[i], prev_coverages[i]
            h, new_state, attn_dist = hyps[i], new_states[i], attn_dists[i]

            # print('h is ', h)
            # print('new_state is ', new_state) shape=(256,)
            # print('attn_dist ids ', attn_dist) shape=(115,)
            # print('p_gen is ', p_gen) 0.4332452
            # print('coverage is ', coverage)shape=(115, 1),
            num += 1
            # print('num is ', num)
            # 分裂 添加 beam size 种可能性
            for j in range(params['beam_size'] * 2):
                # print("++++++++i,j is++++++++",i,j)
                # we extend each hypothesis with each of the top k tokens
                # (this gives 2 x beam_size new hypothesises for each of the beam_size old hypothesises)
                # print('topk_ids is ', topk_ids) shape=(3, 6)
                # print('token is ', topk_log_probs)
                # print('topk_log_probs is ', topk_log_probs)shape=(3, 6)
                # print(topk_ids[i, j].numpy())
                # print('steps is ', steps)
                # print(topk_log_probs[i, j].numpy())
                # print('h is ', h.avg_log_prob)
                # print(coverage)
                # print("topk_ids[i, j].numpy() is",topk_ids[i, j].numpy())
                new_hyp = h.extend(token=topk_ids[i, j].numpy(),
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist
                                   # p_gen=p_gen,
                                   # coverage = new_coverage_i
                                   )
                # print("new_hyp is",new_hyp.tokens)
                # 添加可能情况
                all_hyps.append(new_hyp)

        # in the following lines, we sort all the hypothesises, and select only the beam_size most likely hypothesises
        hyps = []
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        # 筛选top前beam_size句话
        for h in sorted_hyps:
            if h.latest_token == vocab.word_to_id('[STOP]'):
                # 长度符合预期,遇到句尾,添加到结果集
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                # 未到结束 ,添加到假设集
                # print(h.latest_token)
                hyps.append(h)
            # 如果假设句子正好等于beam_size 或者结果集正好等于beam_size 就不在添加
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break
        # print('hyps is ', hyps.)
        # print('steps is ', steps)
        steps += 1

    if len(results) == 0:
        results = hyps

    # At the end of the loop we return the most likely hypothesis, which holds the most likely ouput sequence,
    # given the input fed to the model
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    # print(batch['enc_input'])
    # orig_text = " ".join([vocab.id_to_word(int(index)) for index in batch['enc_input'][0]])
    # print("输入的文本为",orig_text)

    best_hyp = hyps_sorted[0]
    # print('best_hyp.tokens is ', best_hyp.tokens)
    best_hyp.abstract = " ".join([vocab.id_to_word(index) for index in best_hyp.tokens])

    # print('best_hyp.abstract is ', best_hyp.abstract)
    return best_hyp
