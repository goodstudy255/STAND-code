import pandas as pd
import numpy as np
import sys
import math
import random
sys.path.append('/Users/hanzhexin/Desktop/snack_model_config/STAMP/')
from data_prepare.entity.sample import Sample
from data_prepare.entity.sample_kuairand import Sample_kuairand
from data_prepare.entity.samplepack import Samplepack


def load_data_kuairand(data_file, video_file, pad_idx=0, class_num = 3):
    '''
    ret = [contexts, aspects, labels, positions] ,
    context.shape = [len(samples), None], None should be the len(context); 
    aspects.shape = [len(samples), None], None should be the len(aspect);
    labels.shape = [len(samples)]
    positions.shape = [len(samples), 2], the 2 means from and to.
    '''
    # the global param.

    items2idx = {}  # the ret
    items2idx['<pad>'] = pad_idx
    


    idx_cnt = 0
    duration_cnt = 0
    play_cnt = 0

    # load the data
    # idx_cnt  第几个数据
    # train_data, idx_cnt = _load_data(train_file, items2idx, idx_cnt, pad_idx, class_num)
    # print(len(items2idx.keys()))
    # test_data, idx_cnt = _load_data(test_file, items2idx, idx_cnt, pad_idx, class_num)
    train_data, test_data,idx_cnt = _load_data(data_file,video_file, items2idx, idx_cnt, pad_idx, class_num)
    # print(len(items2idx.keys()))
    item_num = len(items2idx.keys())
    return train_data, test_data, items2idx , item_num



def _load_data(file_path,video_file, item2idx, idx_cnt, pad_idx, class_num):
    # data = pd.read_csv()
    # print(file_path)
    data = pd.read_csv(file_path, sep=',', dtype={'itemId': np.int64})
    data = data[data['is_click']==1]
    data = data[data['tab']==1]
    print("read finish")
    # return
    data.sort_values(['user_id','time_ms'],inplace=True)  # 按照user_id,date和hourmin生序排列

    video_data = pd.read_csv(video_file)

    # data.sort_values(['sessionId', 'Time'], inplace=True)  # 按照sessionid和时间升序排列
    print("sort finish")
    # y = list(data.groupby('SessionId'))
    print("list finish")
    # tmp_data = dict(y)


    samplepack_train = Samplepack()
    samplepack_test = Samplepack()

    train_samples = []
    test_samples = []
    
    train_start_id = 0
    test_start_id = 0

    samples = []
    now_id = 0
    print("I am reading")
    sample = Sample()
    last_id = None
    click_items = []
    session_id = 0
    session_length = 50
    recent_length = 1
    history_length  =session_length - recent_length

    
    date_list = []
    is_click_list = []
    label_list = []

    cate_dict = {}
    item2cate = {}
    # for item_id, cate in enumerate(video_data['tag']):
    #     if type(cate) == str:
    #         cate = cate.split(',')[0]
    #         if cate not in cate_dict:
    #             cate_dict[cate] = len(cate_dict) + 1
    #         item2cate[item_id] = cate_dict[cate]
    #     else:
    #         item2cate[item_id] = 0
    item2label = {}
    for video_id, label in zip(list(video_data['video_id'].values),list(video_data['tag'].values)):
        label = label.split(',')[0]
        if video_id not in item2label:
            item2label[video_id] = int(label)

    for u_id,item_id,date in zip(list(data['user_id'].values),list(data['video_id'].values),list(data['date'].values)):
        if last_id is None:
            last_id = u_id
        if u_id != last_id:
            flag =  True
            for i in range(len(date_list)):
                if date_list[i]==20220507 and flag:
                    if i < session_length:
                        pad_zero_num = session_length-i
                        train_start_id = 0
                    else:
                        pad_zero_num = 0
                        train_start_id = i-session_length
                    flag = False
                elif date_list[i] == 20220508:
                    test_start_id = pad_zero_num+i-session_length
                    break

            click_items = [0]*pad_zero_num + click_items
            play_time_list = [0]*pad_zero_num + play_time_list
            duration_time_list = [0]*pad_zero_num + duration_time_list
            date_list = [0]*pad_zero_num + date_list

        
            item_dixes = []
            item_durations = []
            item_plays = []
            # for i in range(len(click_items)):
            #     if i>=train_start_id:
            #         item = click_items[i]
            for item in click_items:
                if item not in item2idx:
                    if idx_cnt == pad_idx:
                        idx_cnt += 1
                    item2idx[item] = idx_cnt
                    idx_cnt += 1
                item_dixes.append(item2idx[item])

            
            # for i in range(len(click_items)):
            #     item = click_items[i]
            #     if i >start_id:
            #         if item not in item2idx:
            #             if idx_cnt == pad_idx:
            #                 idx_cnt += 1
            #             item2idx[item] = idx_cnt
            #             idx_cnt += 1
            #         item_dixes.append(item2idx[item])

            
            # in_dixes = item_dixes[:-1]
            # out_dixes = item_dixes[1:]
            for i in range(train_start_id, len(click_items)-session_length):
                sample.id = now_id
                sample.session_id = last_id
                sample.click_items = click_items[i:i+session_length]
                sample.items_idxes = item_dixes[i:i+session_length]
                
                # sample.in_idxes = item_dixes[i:i+history_length]
                # sample.out_idxes = item_dixes[i+history_length:i+session_length]
                if i<test_start_id:
                    sample.in_idxes = item_dixes[i:i+history_length]
                    sample.out_idxes = item_dixes[i+recent_length:i+session_length]

                    sample.in_label = label_list[i:i+history_length]
                    sample.out_label = label_list[i+recent_length:i+session_length]
                    
                    train_samples.append(sample)

                else:
                    sample.in_idxes = item_dixes[i:i+history_length]
                    sample.out_idxes = item_dixes[i+history_length:i+session_length]
                    
                    sample.in_label = label_list[i:i+history_length]
                    sample.out_label = label_list[i+history_length:i+session_length]
                    
                    test_samples.append(sample)

                
                # if i < test_start_id:
                #     train_samples.append(sample)
                # else:
                #     test_samples.append(sample)
                # samples.append(sample)
            
                sample = Sample_kuairand()
                now_id += 1
                
            last_id =u_id
            click_items = []   
            label_list = []
            date_list = [] 
   
        else:
            last_id = u_id
        click_items.append(item_id)
        label_list.append(item2label[item_id])    
        date_list.append(date)


        # click_items = list(tmp_data[session_tmp_idx]['ItemId'])
    sample = Sample_kuairand()
    item_dixes = []
    item_durations = []
    item_plays = []

    # for i in range(len(click_items)):
    #     if i>=train_start_id:
    #         item = click_items[i]
    for item in click_items:
        if item not in item2idx:
            if idx_cnt == pad_idx:
                idx_cnt += 1
            item2idx[item] = idx_cnt
            idx_cnt += 1
        item_dixes.append(item2idx[item])

    # for i in range(len(duration_time_list)):
    #     if i>=train_start_id:
    #         duration = duration_time_list[i]


    
    print('训练样本数量:',len(train_samples))
    print('测试样本数量:',len(test_samples))

    
    samplepack_train.samples = train_samples
    samplepack_train.init_id2sample()
    samplepack_test.samples = test_samples
    samplepack_test.init_id2sample()
    return samplepack_train, samplepack_test,idx_cnt


if __name__ == '__main__':
    data_path = r'/Users/hanzhexin/Desktop/snack_model_config/STAMP/KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv'
    video_path = r'/Users/hanzhexin/Desktop/snack_model_config/STAMP/KuaiRand-1K/data/video_features_basic_1k.csv'
    # data_path = r'/Users/hanzhexin/Desktop/snack_model_config/STAMP/KuaiRand-27K/data/log_random_4_22_to_5_08_27k.csv'
    items2idx = {}  # the ret
    pad_idx = 0
    items2idx['<pad>'] = pad_idx
    idx_cnt = 0 
    class_num = 5
    train_data,test_data,items2idx, item_num = load_data_kuairand(data_path,video_path,pad_idx,class_num)
    # train_data, idx_cnt = _load_data(data_path, items2idx, idx_cnt, pad_idx, class_num)
    kk=1
