import pandas as pd
import numpy as np
import sys
import math
import random
import json
from collections import defaultdict
from data_prepare.entity.sample import Sample
from data_prepare.entity.sample_kuairand import Sample_kuairand
from data_prepare.entity.samplepack import Samplepack


def load_data_m(train_file, test_file,pad_idx = 0):
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
    items2idx[0] = 0


    path = 'ml-1m/item2cate.json'
    item2tag_float = json.load(open(path))
    item2tag = defaultdict()
    item2tag['<pad>'] = pad_idx
    item2tag[0] = [0]
    tag_list = []
    for item in item2tag_float:
        cate_list = item2tag_float[item].split(',')
        for x in cate_list:
            if int(x) not in tag_list:
                tag_list.append(int(x))            
        item2tag[int(item)] = [int(x) for x in cate_list]
    tag_num = len(tag_list)

    users = {}
    interactions = {}

    tag2idx = {}
    tag2idx['<pad>'] = pad_idx
    idx_cnt = 0

    # load the data
    train_data, idx_cnt = _load_data(train_file, items2idx,item2tag, idx_cnt ,tag_num ,users,interactions,pad_idx)

    test_data, idx_cnt = _load_data(test_file, items2idx, item2tag,idx_cnt,tag_num,users,interactions, pad_idx)

    for item in item2tag:
        cate_list = item2tag[item]
        if cate_list == 0:
            continue
        tag = [0]*tag_num
        for cate in cate_list:
            tag[int(cate)] = 1
        item2tag[item] = tag

    item_num = len(items2idx.keys())
    return train_data, test_data, items2idx, item_num,item2tag,tag_num

def parse_func(line):
    fields = line.split('\t')
    user = int(fields[0])
    pos = int(fields[2])
    tag = fields[3].split(',')
    tag = [int(t) for t in tag]
    seq = np.array([int(x) for x in fields[1].split(',')])
    assert len(seq) == 50
    return (user, seq, pos, tag)



def _load_data(file_path, item2idx, item2tag,idx_cnt,  tag_num, users,interactions, pad_idx=0):
    with open(file_path) as f:
        lines = list(map(parse_func, f.readlines()))

    samplepack = Samplepack()
    samples = []
    now_id = 0
    print("I am reading")
    sample = Sample_kuairand()
    last_id = None
    click_items = []

    for data in lines:
        sample.id = now_id
        sample.session_id = data[0]   
        if data[0] not in users:
            users[data[0]] =1
        item_dixes = []
        tag_list = []
        for item in data[1]:
            if int(item) != 0:
                interaction = str(data[0])+'_'+str(item)
                if interaction not in interactions:
                    interactions[interaction] = 1

            if item not in item2idx:
                if idx_cnt == pad_idx:
                    idx_cnt +=1
                item2idx[item] = idx_cnt
                idx_cnt+=1
            item_dixes.append(item2idx[item])
            cate_list = item2tag[item]
            tag = [0]*tag_num
            for cate in cate_list:
                tag[int(cate)] = 1
            tag_list.append(tag)
        sample.in_idxes = item_dixes[:50]

        sample.in_tag = tag_list[:50]

        if data[2] not in item2idx:
            item2idx[data[2]] = idx_cnt
            cate_list = item2tag[data[2]]
            tag = [0]*tag_num
            for cate in cate_list:
                tag[int(cate)] = 1
            tag_list.append(tag)
            idx_cnt+=1
        sample.out_idxes = [item2idx[data[2]]]
        sample.out_tag = [tag_list[-1]]
        item_dixes.append(item2idx[data[2]])
        sample.items_idxes = item_dixes

        sample.click_items = np.append(data[1],data[2])
        samples.append(sample)
        now_id+=1
        sample = Sample_kuairand()

    print(len(samples))
    samplepack.samples = samples
    samplepack.init_id2sample()
    return samplepack, idx_cnt


if __name__ == '__main__':

    path = 'ml_1m/item2cate.json'
    item2tag = json.load(open(path))
    a = defaultdict()
    for item in item2tag:
        a[int(item)] = int(item2tag[item])