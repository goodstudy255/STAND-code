# coding=utf-8
from optparse import OptionParser
import tensorflow as tf
import pandas as pd
import numpy as np
from data_prepare.entity.samplepack import Samplepack
from data_prepare.load_dict import load_random,load_random_k,load_random_tag
from data_prepare.kuairand_read_txt import load_data_k
from data_prepare.ml_1m_read_text import load_data_m
from util.Config import read_conf
from util.FileDumpLoad import dump_file, load_file
from util.Randomer import Randomer
import sys
import time
import datetime
import os

# _new
kuairand_train = 'kuairand/kuairand-train_0.txt'
kuairand_test = 'kuairand/kuairand-test_0.txt'

ml_1m_train = 'ml-1m/ml-1m-train_0.txt'
ml_1m_test = 'ml-1m-test_0.txt'

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

            emb_dict_id = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            emb_dict_tag  = load_random_k(max_tag_num,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding_id'] = emb_dict_id
            config['pre_embedding_tag'] = emb_dict_tag
            config['item2tag'] = item2tag
            config['item2idx'] = item2idx

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



    else:
        print ("not reload the datasets.")
        print(config['dataset'])

        if config['dataset'] == 'kuairand':
            train_data, test_data, item2idx,n_items,max_num,item2tag,max_tag_num = load_data_k(
                kuairand_train,
                kuairand_test
            )
            config["n_items"] = n_items-1
            emb_dict_id = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])     
            # emb_dict_tag  = load_random(item2tag,edim=config['hidden_size'], init_std=config['emb_stddev'])
            emb_dict_tag = load_random_k(max_tag_num,edim=config['hidden_size'], init_std=config['emb_stddev'])

            config['pre_embedding_id'] = emb_dict_id
            config['pre_embedding_tag'] = emb_dict_tag
            config['item2idx'] = item2idx

            config['item2tag'] = item2tag

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
        default='stand_ml_1m'
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
        default='./save_model'
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

    train_model_save_path = 'save_model/stamp_new/save3/ml_1m.ckpt-kuairand'

    # setup randomer

    Randomer.set_stddev(config['stddev'])
    # 只有测试的话只测试一次
    if not is_train and test_data!=None:
        max_recall = []
        max_mrr = []
        cut_off = [1,2,5,20,50]
        # cut_off = [5,10,50,100,200]
        epoch_num = 0
        for i in range(len(cut_off)):
            max_recall.append(0.0)
            max_mrr.append(0.0)
        test_graph = tf.Graph()
        with test_graph.as_default() as g2:
            saver = tf.train.Saver()
            model = getattr(module, obj)(config)
            model.build_test_model() 
            with tf.Session(graph=g2) as test_sess:    
                test_sess.run(tf.global_variables_initializer())
                sent_data = test_data
                saver.restore(test_sess, train_model_save_path)
                recall, mrr = model.test(test_sess, sent_data)
                print(recall, mrr)
                increase_num = 0
                for i in range(len(max_recall)):
                    if max_recall[i] < recall[i]:
                        max_recall[i] = recall[i]
                        test_data.update_best()
                        increase_num+=1
                        epoch_num = 0
                    if max_mrr[i] <mrr[i]:
                        max_mrr[i] = mrr[i]
                        test_data.update_best()
                        increase_num+=1
                        epoch_num = 0
                    if max_recall[i] > config['kuairand_threshold_acc']:
                        model.save_model(test_sess, config, saver)    
                    print ("max_recall@{}: ".format(str(cut_off[i])) + str(max_recall[i])+" max_mrr@{}: ".format(str(cut_off[i]))+str(max_mrr[i]))
                if increase_num==0:
                    epoch_num += 1
                if epoch_num==3:
                    print("长时间指标未增长，训练结束")
                    sys.exit(0)    
                test_data.flush()
    else:
        train_graph = tf.Graph()
        with train_graph.as_default():
            # build model
            train_model = getattr(module, obj)(config)
            train_model.build_train_model()
        test_graph = tf.Graph()
        with test_graph.as_default():
            test_model = getattr(module, obj)(config)
            test_model.build_test_model()
        with tf.Session(graph =train_graph) as train_sess:
            train_sess.run(tf.global_variables_initializer())

            Total_params = 0 
            Trainable_params = 0
            NonTrainable_params = 0

            for var in tf.global_variables():
                shape = var.shape 
                array = np.asarray([dim.value for dim in shape]) 
                mulValue = np.prod(array) 

                Total_params += mulValue 
                if var.trainable:
                    Trainable_params += mulValue 
                else:
                    NonTrainable_params += mulValue

            print(f'Total params: {Total_params}')
            print(f'Trainable params: {Trainable_params}')
            print(f'Non-trainable params: {NonTrainable_params}')

            merged = tf.summary.merge_all() 
            log_path = "loss_log/"+str(datetime.datetime.now())
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            writer = tf.summary.FileWriter(log_path, train_sess.graph)
            max_recall = []
            max_mrr = []
            cut_off = [1,2,5,20,50]
            epoch_num = 0
            for i in range(len(cut_off)):
                max_recall.append(0.0)
                max_mrr.append(0.0)
            for e in range(epoch):
                if is_train:
                    if dataset == 'kuairand' or dataset == 'ml_1m':
                        train_model.train(train_sess,e, train_data, merged, writer,  threshold_acc=config['kuairand_threshold_acc'])
                    else:
                        train_model.train(train_sess, train_data, test_data, threshold_acc=config['recsys_threshold_acc'])
                    print('训练时间',time.time()-start)
                    train_saver = tf.train.Saver()  
                    train_saver.save(train_sess, train_model_save_path)
                if test_data != None:
                    with tf.Session(graph=test_graph) as test_sess:
                        test_sess.run(tf.global_variables_initializer())    
                        test_saver = tf.train.Saver() 
                        test_saver.restore(test_sess, train_model_save_path)
                        sent_data = test_data
                        start = time.time()
                        recall, mrr = test_model.test(test_sess, sent_data)
                        print('测试时间',time.time()-start)
                        print(recall, mrr)
                        increase_num = 0
                        for i in range(len(max_recall)):
                            if max_recall[i] < recall[i]:
                                max_recall[i] = recall[i]
                                test_data.update_best()
                                increase_num+=1
                                epoch_num = 0
                            if max_mrr[i] <mrr[i]:
                                max_mrr[i] = mrr[i]
                                test_data.update_best()
                                increase_num+=1
                                epoch_num = 0
                            if max_recall[i] > config['kuairand_threshold_acc']:
                                test_model.save_model(test_sess, config, test_saver)    
                            print ("max_recall@{}: ".format(str(cut_off[i])) + str(max_recall[i])+" max_mrr@{}: ".format(str(cut_off[i]))+str(max_mrr[i]))
                        if increase_num==0:
                            epoch_num += 1
                        if epoch_num==5:
                            sys.exit(0)    
                        test_data.flush()

if __name__ == '__main__':
    options = option_parse()
    main(options)
    