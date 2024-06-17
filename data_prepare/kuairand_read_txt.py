import pandas as pd
import numpy as np
import sys
import math
import random
import json
from collections import defaultdict
from data_prepare.entity.sample import Sample
from data_prepare.entity.samplepack import Samplepack


def load_data_k(train_file, test_file,pad_idx = 0):
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


    path = 'kuairand/item2cate.json'
    item2tag_float = json.load(open(path))
    item2tag = defaultdict()
    item2tag['<pad>'] = pad_idx
    item2tag[0] = 0
    max_tag_num = 0
    for item in item2tag_float:
        item2tag[int(item)] = int(item2tag_float[item])
        max_tag_num = max(item2tag[int(item)],max_tag_num)

    tag2idx = {}
    tag2idx['<pad>'] = pad_idx
    max_num = 0

    users = {}
    interactions = {}
    

    idx_cnt = 0
    # load the data
    train_data, idx_cnt,max_num = _load_data(train_file, items2idx,item2tag, idx_cnt,max_num,users,interactions,pad_idx)
    test_data, idx_cnt,max_num = _load_data(test_file, items2idx, item2tag,idx_cnt,max_num,users,interactions, pad_idx)

    item_num = len(items2idx.keys())
    return train_data, test_data, items2idx, item_num,max_num,item2tag,max_tag_num

def parse_func(line):
    fields = line.split('\t')
    user = int(fields[0])
    pos = int(fields[2])
    tag = int(fields[3])
    seq = np.array([int(x) for x in fields[1].split(',')])
    assert len(seq) == 50
    return (user, seq, pos, tag)

def _load_data(file_path, item2idx, item2tag,idx_cnt,max_num, users,interactions, pad_idx=0):

    with open(file_path) as f:
        lines = list(map(parse_func, f.readlines()))

    samplepack = Samplepack()
    samples = []
    now_id = 0
    print("I am reading")
    sample = Sample()
    last_id = None
    click_items = []

    for data in lines:
        sample.id = now_id
        sample.session_id = data[0]
        if data[0] not in users:
            users[data[0]] =1
        sample.in_tag = data[3]  
        item_dixes = []
        tag_list = []
        tag_dixes = []
        max_value = max(data[1])
        max_num = max(max_num,max_value,data[3])
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
            tag_list.append(item2tag[item])
        sample.in_idxes = item_dixes[:50]

        sample.in_tag = tag_list[:50]

        if data[2] not in item2idx:
            item2idx[data[2]] = idx_cnt
            tag_list.append(item2tag[data[2]])
            idx_cnt+=1
        sample.out_idxes = [item2idx[data[2]]]
        sample.out_tag = [tag_list[-1]]
        item_dixes.append(item2idx[data[2]])
        sample.items_idxes = item_dixes

        sample.click_items = np.append(data[1],data[2])
        samples.append(sample)
        now_id+=1
        sample = Sample()

    print(len(samples))
    samplepack.samples = samples
    samplepack.init_id2sample()
    return samplepack, idx_cnt,max_num


if __name__ == '__main__':

    path = 'kuairand/item2cate.json'
    item2tag = json.load(open(path))
    a = defaultdict()
    for item in item2tag:
        a[int(item)] = int(item2tag[item])