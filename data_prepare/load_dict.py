import numpy as np
import copy

def load_random( word2idx, pad_idx=0, edim=300, init_std=0.05):
    emb_dict = np.random.normal(0, init_std, [len(word2idx), edim])
    emb_dict[pad_idx] = [0.0] * edim
    return emb_dict


def load_random_k( max_num, pad_idx=0, edim=300, init_std=0.05):  
    emb_dict = np.random.normal(0, init_std, [max_num, edim])
    emb_dict[pad_idx] = [0.0] * edim
    return emb_dict

def load_random_tag( max_tag_num,max_num,item_data,item2tag, pad_idx=0, edim=300, init_std=0.05):
    emb_dict = np.random.normal(0, init_std, [max_tag_num+1, edim])
    emb_new_dict = []
    emb_dict[pad_idx] = [0.0] * edim
    item2tag[0] = 0
    for item in item_data:
        if item == '<pad>':
            continue
        tag = item2tag[int(item)]
        emb_new_dict.append(emb_dict[tag].tolist())
    if len(emb_new_dict)<max_num+1:
        emb_new_dict+=[[0]*100]*(max_num+1-len(emb_new_dict))
    emb_new_dict = np.array(emb_new_dict)
    return emb_dict,emb_new_dict


