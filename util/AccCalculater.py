import time
import math
def cau_acc(pred, labels):
    '''
    Calculate the accuracy. 
    pred.shape = [batch_size]
    pred: the predict labels. 

    labels.shape = [batch_size]
    labels: the gold labels. 
    '''
    acc = 0.0
    wrong_ids = []
    for i in range(len(labels)):
        if labels[i] == pred[i]:
            acc += 1.0
        else:
            wrong_ids.append(i)
    acc /= len(labels)
    return acc, wrong_ids

def cau_samples_acc(samples):
    acc = 0.0
    for sample in samples:
        if sample.is_pred_right():
            acc += 1.0
    acc /= len(samples)
    return acc

def cau_recall_mrr(preds,labels,cutoff):
    recall = []
    mrr = []
    for batch, b_label in zip(preds,labels):
        for step, s_label in zip(batch,b_label):
            ranks = (step[s_label] < step).sum() +1
            recall.append(ranks<=cutoff)
            mrr.append(1/ranks if ranks <= cutoff else 0.0)
    return recall, mrr

def cau_recall_mrr_org(preds,labels,cutoff = 20):
    recall = []
    mrr = []
    rank_l = []

    for batch, b_label in zip(preds,labels):
        ranks = (batch[b_label] < batch).sum() +1
        rank_l.append(ranks)
        recall.append(ranks <= cutoff)
        mrr.append(1/ranks if ranks <= cutoff else 0.0)

    return recall, mrr, rank_l


def cau_recall_mrr_org_list(preds,labels,cutoff = [20]):
    recall = []
    mrr = []
    ndcg = []
    rank_l = []
    for i in range(len(cutoff)):
        recall.append([])
        mrr.append([])
        ndcg.append([])
    for batch, b_label in zip(preds,labels):
        ranks = (batch[b_label] < batch).sum() +1
        rank_l.append(ranks)
        for i in range(len(cutoff)):
            recall[i].append(ranks <= cutoff[i])
            mrr[i].append(1/ranks if ranks <= cutoff[i] else 0.0)
            ndcg[i].append(1/math.log2(ranks+1) if ranks <= cutoff[i] else 0.0)
    return recall, mrr, rank_l

def cau_recall_mrr_n(preds,labels,cutoff = 20):
    recall = []
    mrr = []
    for batch, b_label in zip(preds,labels):

        ranks = (batch[b_label] < batch).sum() +1

        recall.append(ranks <= cutoff)
        mrr.append(1/ranks if ranks <= cutoff else 0.0)
        mrr.append()
    return recall, mrr

def cau_samples_recall_mrr(samples, cutoff=20):
    recall = 0.0
    mrr =0.0
    for sample in samples:
        recall += sum(x <= cutoff for x in sample.pred)
        mrr += sum(1/x if x <= cutoff else 0 for x in sample.pred)
    num = 0
    for sample in samples:
        num += len(sample.pred)
    recall = recall/ num
    mrr = mrr/num
    return recall , mrr

def new_cau_samples_recall_mrr(samples,cutoff=20):
    recall = 0.0
    mrr =0.0
    for sample in samples:
        recall += (1 if sample.pred[0] <= cutoff else 0)
        mrr += (1/sample.pred[0] if sample.pred[0] <=cutoff else 0)
    num = len(samples)
    recall = recall/ num
    mrr = mrr/num
    return recall , mrr





import numpy as np

# change this if using K > 100
denominator_table = np.log2( np.arange( 2, 102 ))

def dcg_at_k( r, k, method = 1 ):
	"""Score is discounted cumulative gain (dcg)

	Relevance is positive real values.  Can use binary
	as the previous methods.

	Example from
	http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
	>>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
	>>> dcg_at_k(r, 1)
	3.0
	>>> dcg_at_k(r, 1, method=1)
	3.0
	>>> dcg_at_k(r, 2)
	5.0
	>>> dcg_at_k(r, 2, method=1)
	4.2618595071429155
	>>> dcg_at_k(r, 10)
	9.6051177391888114
	>>> dcg_at_k(r, 11)
	9.6051177391888114

	Args:
		r: Relevance scores (list or numpy) in rank order
			(first element is the first item)
		k: Number of results to consider
		method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
				If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

	Returns:
		Discounted cumulative gain
	"""
	r = np.asfarray(r)[:k]
	if r.size:
		if method == 0:
			return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
		elif method == 1:
			# return np.sum(r / np.log2(np.arange(2, r.size + 2)))
			return np.sum(r / denominator_table[:r.shape[0]])
		else:
			raise ValueError('method must be 0 or 1.')
	return 0.
 
 
def get_ndcg( r, k, method = 1 ):
	"""Score is normalized discounted cumulative gain (ndcg)

	Relevance orignally was positive real values.  Can use binary
	as the previous methods.

	Example from
	http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
	>>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
	>>> ndcg_at_k(r, 1)
	1.0
	>>> r = [2, 1, 2, 0]
	>>> ndcg_at_k(r, 4)
	0.9203032077642922
	>>> ndcg_at_k(r, 4, method=1)
	0.96519546960144276
	>>> ndcg_at_k([0], 1)
	0.0
	>>> ndcg_at_k([1], 2)
	1.0

	Args:
		r: Relevance scores (list or numpy) in rank order
			(first element is the first item)
		k: Number of results to consider
		method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
				If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

	Returns:
		Normalized discounted cumulative gain
	"""
	dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
	dcg_min = dcg_at_k(sorted(r), k, method)
	#assert( dcg_max >= dcg_min )
	
	if not dcg_max:
		return 0.
	 
	dcg = dcg_at_k(r, k, method)
	
	#print dcg_min, dcg, dcg_max
	
	return (dcg - dcg_min) / (dcg_max - dcg_min)

# ndcg with explicitly given best and worst possible relevances
# for recommendations including unrated movies
def get_ndcg_2( r, best_r, worst_r, k, method = 1 ):

	dcg_max = dcg_at_k( sorted( best_r, reverse = True ), k, method )
	
	if worst_r == None:
		dcg_min = 0.
	else:
		dcg_min = dcg_at_k( sorted( worst_r ), k, method )
		
	# assert( dcg_max >= dcg_min )
	
	if not dcg_max:
		return 0.
	 
	dcg = dcg_at_k( r, k, method )
	
	#print dcg_min, dcg, dcg_max
	
	return ( dcg - dcg_min ) / ( dcg_max - dcg_min )