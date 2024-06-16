# coding=utf-8
from optparse import OptionParser
import tensorflow as tf
import pandas as pd
import numpy as np
from data_prepare.entity.samplepack import Samplepack
from data_prepare.load_dict import load_random,load_random_k,load_random_tag
from data_prepare.cikm16data_read import load_data2
from data_prepare.kuairand_read_txt import load_data_k
from data_prepare.ml_1m_read_text import load_data_m
from data_prepare.rsyc15data_read_p import load_data_p
from util.Config import read_conf
from util.FileDumpLoad import dump_file, load_file
from util.Randomer import Randomer
# the data path.


kuairand_train = 'kuairand/kuairand-train_0.txt'
kuairand_test = 'kuairand/kuairand-test_0.txt'

ml_1m_train = 'ml-1m/ml-1m-train_0.txt'
ml_1m_test = 'ml-1m/ml-1m-test_0.txt'



def load_tt_datas(config={}, reload=True):
    '''
    loda data.
    config: 获得需要加载的数据类型，放入pre_embedding.
    nload: 是否重新解析原始数据
    '''

    if reload:
        print( "reload the datasets.")
        print (config['dataset'])
                #    kuairand_test,
        if config['dataset'] == 'kuairand':
            train_data, test_data, item2idx, n_items,max_num,item2tag,max_tag_num = load_data_k(
                kuairand_train,
                kuairand_test
            )
            config["n_items"] = n_items-1
            # emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])

            emb_dict_id = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            # emb_dict_tag,emb_dict_tag2id = load_random_tag(max_tag_num,max_num,item2idx,item2tag,edim=config['hidden_size'], init_std=config['emb_stddev'])
            emb_dict_tag  = load_random_k(max_tag_num,edim=config['hidden_size'], init_std=config['emb_stddev'])
            # emb_dict_duration_time = load_random(items2duration,edim=config['hidden_size'], init_std=config['emb_stddev'])
            # emb_dict_play_time = load_random(items2play,edim=config['hidden_size'], init_std=config['emb_stddev'])

            # config['pre_embedding'] = emb_dict

            config['pre_embedding_id'] = emb_dict_id
            config['pre_embedding_tag'] = emb_dict_tag
            config['item2tag'] = item2tag
            config['item2idx'] = item2idx
            # config['max_num'] = max_num
            # config['pre_embedding_tag2id'] = emb_dict_tag2id
            # config['pre_embedding_duration_time'] = emb_dict_duration_time
            # config['pre_embedding_play_time'] = emb_dict_play_time

            path = 'datas/mid_data'
            print("-----")

        if config['dataset'] == 'ml_1m':
            train_data, test_data, item2idx, n_items,item2tag,tag_num = load_data_m(
                ml_1m_train,
                ml_1m_test
            )
            config["n_items"] = n_items-1
            emb_dict_id = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            emb_dict_tag  = load_random_k(tag_num,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding_id'] = emb_dict_id
            config['pre_embedding_tag'] = emb_dict_tag
            config['item2tag'] = item2tag
            config['item2idx'] = item2idx
            path = 'datas/mid_data'
            print("-----")


    else:
        print ("not reload the datasets.")
        print(config['dataset'])

        if config['dataset'] == 'kuairand':
            train_data, test_data, item2idx,n_items,max_num,item2tag,max_tag_num = load_data_k(
                kuairand_train,
                kuairand_test
            )
            config["n_items"] = n_items-1
            # emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            # config['pre_embedding'] = emb_dict

            emb_dict_id = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            # emb_dict_tag,emb_dict_tag2id = load_random_tag(max_tag_num,max_num,item2idx,item2tag,edim=config['hidden_size'], init_std=config['emb_stddev'])
            emb_dict_tag  = load_random_k(max_tag_num,edim=config['hidden_size'], init_std=config['emb_stddev'])


            # emb_dict_duration_time = load_random(items2duration,edim=config['hidden_size'], init_std=config['emb_stddev'])
            # emb_dict_play_time = load_random(items2play,edim=config['hidden_size'], init_std=config['emb_stddev'])

            # config['pre_embedding_tag2id'] = emb_dict_tag2id
            config['pre_embedding_id'] = emb_dict_id
            config['pre_embedding_tag'] = emb_dict_tag
            config['item2idx'] = item2idx
            # config['max_num'] = max_num
            config['item2tag'] = item2tag
            # config['pre_embedding_duration_time'] = emb_dict_duration_time
            # config['pre_embedding_play_time'] = emb_dict_play_time

            path = 'datas/mid_data'
            print("-----")

        if config['dataset'] == 'ml_1m':
            train_data, test_data, item2idx, n_items,item2tag,tag_num = load_data_m(
                ml_1m_train,
                ml_1m_test
            )
            config["n_items"] = n_items-1
            emb_dict_id = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            emb_dict_tag  = load_random_k(tag_num,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding_id'] = emb_dict_id
            config['pre_embedding_tag'] = emb_dict_tag
            config['item2tag'] = item2tag
            config['item2idx'] = item2idx
            path = 'datas/mid_data'
            print("-----")

    return train_data, test_data


def load_conf(model, modelconf):
    '''
    model: 需要加载的模型
    modelconf: model config文件所在的路径
    '''
    # load model config
    model_conf = read_conf(model, modelconf)
    if model_conf is None:
        raise Exception("wrong model config path.", model_conf)
    module = model_conf['module']
    obj = model_conf['object']
    params = model_conf['params']
    params = params.split("/")
    paramconf = ""
    model = params[-1]
    for line in params[:-1]:
        paramconf += line + "/"
    paramconf = paramconf[:-1]
    # load super params.
    param_conf = read_conf(model, paramconf)
    return module, obj, param_conf


def option_parse():
    '''
    parse the option.
    '''
    parser = OptionParser()
    parser.add_option(
        "-m",
        "--model",
        action='store',
        type='string',
        dest="model",
        default='stosa_ml_1m'
    )
    parser.add_option(
        "-d",
        "--dataset",
        action='store',
        type='string',
        dest="dataset",
        default='ml_1m'
    )
    parser.add_option(
        "-r",
        "--reload",
        action='store_true',
        dest="reload",
        default=False
    )
    parser.add_option(
        "-c",
        "--classnum",
        action='store',
        type='int',
        dest="classnum",
        default=2
    )

    parser.add_option(
        "-a",
        "--nottrain",
        action='store_true',
        dest="not_train",
        default=False
    )
    parser.add_option(
        "-n",
        "--notsavemodel",
        action='store_true',
        dest="not_save_model",
        default=True
    )
    parser.add_option(
        "-p",
        "--modelpath",
        action='store',
        type='string',
        dest="model_path",
        default='./save_model/caser_model'
    )
    parser.add_option(
        "-i",
        "--inputdata",
        action='store',
        type='string',
        dest="input_data",
        default='test'
    )
    parser.add_option(
        "-e",
        "--epoch",
        action='store',
        type='int',
        dest="epoch",
        default=1
    )
    (option, args) = parser.parse_args()
    return option


def main(options, modelconf="config/model.conf"):
    '''
    model: 需要加载的模型
    dataset: 需要加载的数据集
    reload: 是否需要重新加载数据，yes or no
    modelconf: model config文件所在的路径
    class_num: 分类的类别
    use_term: 是否是对aspect term 进行分类
    '''
    model = options.model
    dataset = options.dataset
    reload = options.reload
    class_num = options.classnum
    is_train = not options.not_train
    is_save = not options.not_save_model
    model_path = options.model_path
    input_data = options.input_data
    epoch = options.epoch

    module, obj, config = load_conf(model, modelconf)
    config['model'] = model
    print(model)
    config['dataset'] = dataset
    config['class_num'] = class_num
    config['nepoch'] = epoch
    train_data, test_data = load_tt_datas(config, reload)
    module = __import__(module, fromlist=True)

    # setup randomer

    Randomer.set_stddev(config['stddev'])

    with tf.Graph().as_default():
        # build model
        model = getattr(module, obj)(config)
        model.build_model()
        if is_save or not is_train:
            saver = tf.train.Saver() #max_to_keep=30
        else:
            saver = None
        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            Total_params = 0 
            Trainable_params = 0
            NonTrainable_params = 0

            # 遍历tf.global_variables()返回的全局变量列表
            for var in tf.global_variables():
                shape = var.shape # 获取每个变量的shape，其类型为'tensorflow.python.framework.tensor_shape.TensorShape'
                array = np.asarray([dim.value for dim in shape]) # 转换为numpy数组，方便后续计算
                mulValue = np.prod(array) # 使用numpy prod接口计算数组所有元素之积

                Total_params += mulValue # 总参数量
                if var.trainable:
                    Trainable_params += mulValue # 可训练参数量
                else:
                    NonTrainable_params += mulValue # 非可训练参数量

            print(f'Total params: {Total_params}')
            print(f'Trainable params: {Trainable_params}')
            print(f'Non-trainable params: {NonTrainable_params}')


            merged = tf.summary.merge_all() 
            writer = tf.summary.FileWriter("loss_log/", sess.graph)
            if is_train:
                print(dataset)
                if dataset == 'kuairand' or dataset == 'ml_1m':
                    model.train(sess, train_data, test_data,merged,writer, saver)
                    saver.save(sess, model_path)
                else:
                    model.train(sess, train_data, test_data, saver, threshold_acc=config['recsys_threshold_acc'])
                # if dataset == "rsc15":
                #     model.train(sess, train_data, test_data, saver, threshold_acc=config['recsys_threshold_acc'])

            else:
                if input_data is "test":
                    sent_data = test_data
                elif input_data is "train":
                    sent_data = train_data
                else:
                    sent_data = test_data
                saver.restore(sess, model_path)
                model.test(sess, sent_data)


if __name__ == '__main__':
    options = option_parse()
    main(options)
    