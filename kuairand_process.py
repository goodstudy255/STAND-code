import os
import json
import numpy as np
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

if not os.path.exists('./kuairand'):
    os.mkdir('./kuairand')
if not os.path.exists('./kuairand_train'):
    os.mkdir('./kuairand_train')
if not os.path.exists('./kuairand_test'):
    os.mkdir('./kuairand_test')

data = pd.read_csv("./KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part1.csv")
data2 = pd.read_csv("./KuaiRand-27K/data/log_standard_4_08_to_4_21_27k_part2.csv")
video_feature = pd.read_csv("./KuaiRand-27K/data/video_features_basic_27k.csv")

stack_data = pd.concat([data, data2], axis=0)

stack_data = stack_data[stack_data['is_click'] == 1]
stack_data = stack_data[stack_data['tab'] == 1]

cnter = Counter(list(stack_data['video_id']))

cate_dict = {}
item2cate = {}
for item_id, cate in enumerate(video_feature['tag']):
    if type(cate) == str:
        cate = cate.split(',')[0]
        if cate not in cate_dict:
            cate_dict[cate] = len(cate_dict) + 1
        item2cate[item_id] = cate_dict[cate]
    else:
        item2cate[item_id] = 0
        
print('itemnum:', len(item2cate))

items_set = set([i for i in cnter])
print(len(items_set))

num = 1

user_list = []
item_list = []
for uid_, vid_ in tqdm(zip(list(stack_data.user_id), list(stack_data.video_id)), total=len(stack_data)):
    if vid_ in items_set:
        user_list.append(uid_)
        item_list.append(vid_)

with open('kuairand_ui_pairs_sorted_by_timestamp.tsv', 'w+') as f:
    for uid_, vid_ in zip(user_list, item_list):
        f.write('{}\t{}\n'.format(uid_, vid_))

import json
data = open('./kuairand/data.txt', 'w+')
user_seq = defaultdict(list)
item_seq = defaultdict(set)
for uid_, vid_ in zip(user_list, item_list):
    user_seq[uid_].append(vid_)
    item_seq[vid_].add(uid_)

user_dict = {}
item_dict = {}
cate_hist = []
cate_dict = {}
itemidx2cate = {}
for uid_ in user_seq:
    if len(user_seq[uid_]) <= 10:
        continue
    if uid_ not in user_dict:
            user_dict[uid_] = len(user_dict) + 1
    for vid_ in user_seq[uid_]:
        if vid_ not in item_dict:
            item_dict[vid_] = len(item_dict) + 1 
        if item2cate[vid_] not in cate_dict:
            cate_dict[item2cate[vid_]] = len(cate_dict) + 1  
        itemidx2cate[str(item_dict[vid_])] = str(cate_dict[item2cate[vid_]])
        data.write('{} {}\n'.format(user_dict[uid_], item_dict[vid_]))

        cate_hist.append(itemidx2cate[str(item_dict[vid_])])
cate_cnter = Counter(cate_hist)
print(cate_cnter)
json.dump(itemidx2cate, open('./kuairand/item2cate.json', 'w')) 
    

import random
data = open('./kuairand/data.txt')
user_seq = defaultdict(list)
item_seq = defaultdict(set)
aver_seq_num = 0
for data_i in data:
    u, i = data_i.strip('\n').split()
    user_seq[u].append(i)
    item_seq[i].add(u)
json.dump(user_seq, open('./kuairand/data.json', 'w'))   

for user_i in user_seq:
    aver_seq_num+=len(user_seq[user_i])
print(aver_seq_num/len(user_seq))
    
rnd_user = list(user_seq.keys())
random.shuffle(rnd_user)
usernum = len(rnd_user)

test_idx = rnd_user[:usernum//5]
train_idx = rnd_user[usernum//5:]
print(len(train_idx))
print(len(test_idx))

test_u_seq = {}
train_u_seq = {}
for u in test_idx:
    if len(user_seq[u]) <= 10:
        continue
    test_u_seq[u] = user_seq[u]

for u in train_idx:
    if len(user_seq[u]) <= 10:
        continue
    train_u_seq[u] = user_seq[u]

print(len(train_u_seq))
print(len(test_u_seq))
json.dump(train_u_seq, open('./kuairand/train_data.json', 'w'))
json.dump(test_u_seq, open('./kuairand/test_data.json', 'w'))

train_data = json.load(open('./kuairand/train_data.json'))
test_data = json.load(open('./kuairand/test_data.json'))
  

usernum = len(train_data)
itemnum_train = []
itemnum_test = []
user_train = {}
user_test = {}
for u in train_data:
    itemnum_train+=train_data[u]
    user_train[int(u)] = [int(i) for i in train_data[u]]
itemnum_train = len(set(itemnum_train))

for u in test_data:
    itemnum_test+=test_data[u]
    user_test[int(u)] = [int(i) for i in test_data[u]]
itemnum_test = len(set(itemnum_test))


item2cate = json.load(open('./kuairand/item2cate.json'))
cate_pool = defaultdict(list)
for item in item2cate:
    item2cate[item] = int(item2cate[item])
    cate = int(item2cate[item])
    cate_pool[cate].append(int(item))
cate = (item2cate, cate_pool)

from multiprocessing import Pool
import os, time, random
import json
def run(part_id, total_part_num):
    print("Starting part " + str(part_id), len(user_train))
    num_items = len(item2cate)
    num_users = len(user_train)
    num_per_usr = 3
    with open('./kuairand-train_%d.txt' % part_id, 'w+') as f_train:
        train_users = list(user_train.keys())
        random.shuffle(train_users)
        for ind, uid in enumerate(train_users):
            seq = [str(i) for i in user_train[uid]]
            seq_set = set(seq)
            seq_len = len(seq)
            if seq_len>100:
                start_id = 100
            else:
                start_id = seq_len
            if seq_len-start_id<num_per_usr:
                num = 1 if seq_len<=start_id else seq_len-start_id
                if num == 1:
                    start_id_list = [start_id]
                else:
                    start_id_list = [k for k in range(start_id,seq_len)]
            else:
                num = num_per_usr
                start_id_list = random.sample(range(start_id,seq_len),num)
            for i in start_id_list:
                i=i-1
                curr_seq_str = ",".join((['0'] * (100-i) + seq[:i]) if i < 100 else seq[i-100:i])
                neg = np.random.randint(itemnum_train)
                tag = item2cate[str(seq[i])]
                cat_list = cate_pool[tag]
                t = np.random.randint(len(cat_list))
                hard_neg = cat_list[t]

                f_train.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (uid, curr_seq_str, seq[i], tag, neg, hard_neg))

        # /kuairand_test_new
    num_per_usr = 1
    with open('./kuairand-test_%d.txt' % part_id, 'w+') as f_test:    
        test_users = list(user_test.keys())
        random.shuffle(test_users)
        for ind, uid in enumerate(test_users):
            seq = [str(i) for i in user_test[uid]]
            seq_set = set(seq)
            seq_len = len(seq)
            if seq_len>100:
                start_id = 100
            else:
                start_id = seq_len

            if seq_len-start_id<num_per_usr:
                num = 1 if seq_len<=start_id else seq_len-start_id
                if num == 1:
                    start_id_list = [start_id]
                else:
                    start_id_list = [k for k in range(start_id,seq_len)]
            else:
                num = num_per_usr
                start_id_list = random.sample(range(start_id,seq_len),num)
            for i in start_id_list:
                i=i-1
                curr_seq_str = ",".join((['0'] * (100-i) + seq[:i]) if i < 100 else seq[i-100:i])
                neg = np.random.randint(itemnum_test)
                tag = item2cate[str(seq[i])]
                cat_list = cate_pool[tag]
                t = np.random.randint(len(cat_list))
                hard_neg = cat_list[t]

                f_test.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (uid, curr_seq_str, seq[i], tag, neg, hard_neg))

        print("Part " + str(part_id) + " finished!")

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(1)
    for i in range(1):  
        p.apply_async(run, args=(i,1))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
