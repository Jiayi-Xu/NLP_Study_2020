import tensorflow as tf
from seq2seq_tf2.models.sequence_to_sequence import SequenceToSequence
from seq2seq_tf2.batcher import batcher, Vocab
from seq2seq_tf2.train_helper import train_model
from seq2seq_tf2.test_helper import  batch_greedy_decode, beam_decode
from tqdm import tqdm
from utils.data_utils import get_result_filename
import pandas as pd
# from rouge import Rouge
import pprint



def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print('true vocab is ', vocab)
    print("建立word和id的索引关系完成")

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Building the model ...")
    model = SequenceToSequence(params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
    ckpt = tf.train.Checkpoint(SequenceToSequence=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Starting the training ...")
    train_model(model, b, params, ckpt, ckpt_manager)


def test(params):
    assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    model = SequenceToSequence(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    # print("Creating the batcher ...")
    # b = batcher(vocab, params) 在predict_result执行的

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
    ckpt = tf.train.Checkpoint(SequenceToSequence=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
    # path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")
    # for batch in b:
    #     yield batch_greedy_decode(model, batch, vocab, params)

    """    
    修改：
        去掉了predict_result 函数
        将处理steps_epoch的共用代码提取出来，再进行分支greedy_decode/beam_decode的出来
    """

    # 调用batcher里的batcher->batch_generator函数 生成器example_generator mode == "test" 207行开始
    dataset = batcher(vocab, params)
    # 测试集的数量
    sample_size = params['sample_size']
    steps_epoch = sample_size // params["batch_size"] + 1
    results = []
    for i in tqdm(range(steps_epoch)):
        enc_data, _ = next(iter(dataset))

        # 如果为TRUE进行贪心搜索 否则BEAM SEARCH
        if params['greedy_decode']:
            # print("-----------------greedy_decode 模式-----------------")
            results += batch_greedy_decode(model, enc_data, vocab, params)
        else:
            # print("-----------------beam_decode 模式-----------------")
            # print(enc_data["enc_input"][0])
            # print(enc_data["enc_input"][1])
            # 需要beam sezi=batch size 输入时候相当于遍历一个个X 去进行搜索
            for row in range(params['batch_size']):
                batch = [enc_data["enc_input"][row] for _ in range(params['beam_size'])]
                best_hyp = beam_decode(model, batch, vocab, params)
                results.append(best_hyp.abstract)


    # batch遍历完成 保存测试结果
    results = list(map(lambda x: x.replace(" ", ""), results))
    # 保存结果 AutoMaster_TestSet.csv
    save_predict_result(results, params)

    # save_predict_result(results, params)
    print('save beam search result to :{}'.format(params['test_x_dir']))

# test 模式
# def predict_result(model, params, vocab):
#     # 调用batcher里的batcher->batch_generator函数 生成器example_generator mode == "test" 207行开始
#     dataset = batcher(vocab, params)
#     # 预测结果
#     # results = greedy_decode(model, dataset, vocab, params)
#     results = greedy_decode(model, dataset, vocab, params)
#
#     results = list(map(lambda x: x.replace(" ",""), results))
#     # 保存结果 AutoMaster_TestSet.csv
#     save_predict_result(results, params)
#
#     return results


def save_predict_result(results, params):
    sample_size = params['sample_size']
    # 读取结果 AutoMaster_TestSet.csv
    test_df = pd.read_csv(params['test_x_dir'])
    # 填充结果
    test_df['Prediction'] = results[:sample_size]
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    result_save_path = get_result_filename(params)
    test_df.to_csv(result_save_path, index=None, sep=',')


def test_and_save(params):
    assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test(params)
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            with open(params["test_save_dir"] + "/article_" + str(i) + ".txt", "w", encoding='utf-8') as f:
                f.write("article:\n")
                f.write(trial.text)
                f.write("\n\nabstract:\n")
                f.write(trial.abstract)
            pbar.update(1)


def evaluate(params):
    gen = test(params)
    reals = []
    preds = []
    with tqdm(total=params["max_num_to_eval"], position=0, leave=True) as pbar:
        for i in range(params["max_num_to_eval"]):
            trial = next(gen)
            reals.append(trial.real_abstract)
            preds.append(trial.abstract)
            pbar.update(1)
    r = Rouge()
    scores = r.get_scores(preds, reals, avg=True)
    print("\n\n")
    pprint.pprint(scores)


if __name__ == '__main__':
    pass

