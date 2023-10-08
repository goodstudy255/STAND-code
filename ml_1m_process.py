import os
import numpy as np
from collections import defaultdict
import json
import pandas as pd

# ml_1m_data_path = '/Users/hanzhexin/Desktop/STAMP/ml-1m/ratings.dat'

# user_seq = defaultdict(list)
# item_seq = defaultdict(set)
# with open(ml_1m_data_path,'r', encoding='utf-8') as f:
#     for rating in f.readlines():
#         rating = rating.strip('\n')
#         rating = rating.split('::')
#         usr_id = rating[0]
#         movie_id = rating[1]
#         user_seq[usr_id].append(movie_id)
#         item_seq[movie_id].add(usr_id)

# cate_path = '/Users/hanzhexin/Desktop/STAMP/ml-1m/movies.dat'

# cate_dict = {}
# item2cate = {}
# id = 0
# with open(cate_path,'r',encoding='ISO-8859-1') as f:
#     for line in f.readlines():
#         line = line.strip('\n')
#         line = line.split('::')
#         cate = line[-1]
#         cate_list = []
#         if '|' in cate:
#             cate = cate.split('|')
#             for c in cate:
#                 if c not in cate_dict:
#                     cate_dict[c] = id
#                     id+=1
#                 cate_list.append(cate_dict[c])
#         else:
#             if cate not in cate_dict:
#                 cate_dict[cate] = id
#                 id+=1
#             cate_list.append(cate_dict[cate])
#         movie_id = line[0]
#         # if cate not in cate_dict:
#         #     cate_dict[cate] = id
#         #     id+=1
#         item2cate[movie_id] = cate_list #cate_dict[cate]

# data = open('./ml-1m/data.txt', 'w+')
# user_dict = {}
# item_dict = {}
# itemidx2cate = {}

# for uid_ in user_seq:
#     if len(user_seq[uid_]) <= 5:
#         continue
#     if uid_ not in user_dict:
#         user_dict[uid_] = len(user_dict) + 1
#     for vid_ in user_seq[uid_]:
#         if vid_ not in item_dict:
#             item_dict[vid_] = len(item_dict) + 1  
#         cate = ''
#         for c in item2cate[vid_]:
#             cate += str(c)
#             cate += ','
#         itemidx2cate[str(item_dict[vid_])] = str(cate[:-1])
#         data.write('{} {}\n'.format(user_dict[uid_], item_dict[vid_]))
# json.dump(itemidx2cate, open('./ml-1m/item2cate.json', 'w'))   

# import random
# data = open('./ml-1m/data.txt')
# user_seq = defaultdict(list)
# item_seq = defaultdict(set)
# aver_seq_num = 0
# for data_i in data:
#     u, i = data_i.strip('\n').split()
#     user_seq[u].append(i)
#     item_seq[i].add(u)
# json.dump(user_seq, open('./ml-1m/data.json', 'w'))   

# for user_i in user_seq:
#     aver_seq_num+=len(user_seq[user_i])
# print(aver_seq_num/len(user_seq))
    
# rnd_user = list(user_seq.keys())
# random.shuffle(rnd_user)
# usernum = len(rnd_user)

# test_idx = rnd_user[:usernum//5]
# train_idx = rnd_user[usernum//5:]
# print(len(train_idx))
# print(len(test_idx))

# test_u_seq = {}
# train_u_seq = {}
# for u in test_idx:
#     if len(user_seq[u]) <= 5:
#         continue
#     test_u_seq[u] = user_seq[u]

# for u in train_idx:
#     if len(user_seq[u]) <= 5:
#         continue
#     train_u_seq[u] = user_seq[u]

# print(len(train_u_seq))
# print(len(test_u_seq))
# json.dump(train_u_seq, open('./ml-1m/train_data.json', 'w'))
# json.dump(test_u_seq, open('./ml-1m/test_data.json', 'w')) 


train_data = json.load(open('./ml-1m/train_data.json'))
test_data = json.load(open('./ml-1m/test_data.json'))
  

usernum = len(train_data)
itemnum_train = []
itemnum_test = []
user_train = {}
user_test = {}
for u in train_data:
    itemnum_train+=train_data[u]
    user_train[int(u)] = [int(i) for i in train_data[u]]
# print(itemnum_train, np.max(list(user_train.keys())))
itemnum_train = len(set(itemnum_train))

for u in test_data:
    itemnum_test+=test_data[u]
    user_test[int(u)] = [int(i) for i in test_data[u]]
# print(itemnum_test, np.max(list(user_train.keys())))
itemnum_test = len(set(itemnum_test))



item2cate = json.load(open('./ml-1m/item2cate.json'))
cate_pool = defaultdict(list)
for item in item2cate:
    cate_list = item2cate[item].split(',')
    item2cate[item] = cate_list
    for cate in cate_list:
        cate = int(cate)
        cate_pool[cate].append(int(item))
cate = (item2cate, cate_pool)

from multiprocessing import Pool
import os, time, random
import json

# os.mkdir('./ml-1m-data/')
def run(part_id, total_part_num):
    print("Starting part " + str(part_id), len(user_train))
    num_items = len(item2cate)
    num_users = len(user_train)
    print(num_items)
    num_per_usr = 10
    with open('./ml-1m/ml-1m-train_%d.txt' % part_id, 'w+') as f_train:
        train_users = list(user_train.keys())
        random.shuffle(train_users)
        for ind, uid in enumerate(train_users):
            seq = [str(i) for i in user_train[uid]]
            seq_set = set(seq)
            seq_len = len(seq)
            if seq_len>50:
                start_id = 50
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
                curr_seq_str = ",".join((['0'] * (50-i) + seq[:i]) if i < 50 else seq[i-50:i])
                neg = np.random.randint(itemnum_train)
                tag = item2cate[str(seq[i])]
                cat_list = cate_pool[int(tag[0])]
                s = ''
                for t in tag:
                    s = s+str(t)
                    s+=','
                s = s[:-1]
                tag = s
                # print(tag)
                # print(len(cat_list))
                t = np.random.randint(len(cat_list))
                
                hard_neg = cat_list[t]
                # print(hard_neg)
                

                f_train.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (uid, curr_seq_str, seq[i], tag, neg, hard_neg))

        
    num_per_usr = 1
    with open('./ml-1m/ml-1m-test_%d.txt' % part_id, 'w+') as f_test:    
        test_users = list(user_test.keys())
        random.shuffle(test_users)
        for ind, uid in enumerate(test_users):
            seq = [str(i) for i in user_test[uid]]
            seq_set = set(seq)
            seq_len = len(seq)
            if seq_len>50:
                start_id = 50
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
                curr_seq_str = ",".join((['0'] * (50-i) + seq[:i]) if i < 50 else seq[i-50:i])
                neg = np.random.randint(itemnum_test)
                tag = item2cate[str(seq[i])]
                cat_list = cate_pool[int(tag[0])]
                s = ''
                for t in tag:
                    s = s+str(t)
                    s+=','
                s = s[:-1]
                tag = s
                
                # cat_list = cate_pool[tag]
                t = np.random.randint(len(cat_list))
                hard_neg = cat_list[t]

                f_test.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (uid, curr_seq_str, seq[i], tag, neg, hard_neg))

        print("Part " + str(part_id) + " finished!")


if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(5)
    for i in range(5):   #100为所需要的txt数量。   
        p.apply_async(run, args=(i,5))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')