#coding=utf-8
import numpy as np
import tensorflow as tf
import time
from basic_layer.NN_adam import NN
from util.Printer import TIPrint
from util.batcher.equal_len.batcher_p import batcher
from util.AccCalculater import cau_recall_mrr_org_list
from util.AccCalculater import cau_samples_recall_mrr
from util.Pooler import pooler
from basic_layer.FwNn3AttLayer import FwNnAttLayer
from util.FileDumpLoad import dump_file
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import sys


class Seq2SeqAttNN(NN):
    """
    The memory network with context attention.
    """
    # ctx_input.shape=[batch_size, mem_size]

    def __init__(self, config):
        super(Seq2SeqAttNN, self).__init__(config)
        self.config = None
        if config != None:
            self.config = config
            # the config.
            self.datas = config['dataset']
            self.nepoch = config['nepoch']  # the train epoches.
            self.batch_size = config['batch_size']  # the max train batch size.
            self.init_lr = config['init_lr']  # the initialize learning rate.
            # the base of the initialization of the parameters.
            self.stddev = config['stddev']
            self.edim = config['edim']  # the dim of the embedding.
            self.max_grad_norm = config['max_grad_norm']   # the L2 norm.
            self.n_items = config["n_items"]
            # the pad id in the embedding dictionary.
            self.pad_idx = config['pad_idx']
            self.item2tag = config['item2tag']
            self.item2idx = config['item2idx']

            self.pre_embedding = config['pre_embedding_id']
            self.pre_embedding_tag = config['pre_embedding_tag']
            # self.pre_embedding_tag2id = config['pre_embedding_tag2id']

            # generate the pre_embedding mask.
            self.pre_embedding_mask = np.ones(np.shape(self.pre_embedding))
            self.pre_embedding_mask[self.pad_idx] = 0

            self.pre_embedding_tag_mask = np.ones(np.shape(self.pre_embedding_tag))
            self.pre_embedding_tag_mask[self.pad_idx] = 0

            self.pre_tag = []
            for i in range(len(self.pre_embedding)-1):
                self.pre_tag.append(self.item2tag[i])

            self.emb_up = config['emb_up']

            # the active function.
            self.active = config['active']

            # hidden size
            self.hidden_size = config['hidden_size']

            self.is_print = config['is_print']

            self.cut_off = config["cut_off"]
        self.is_first = True
        # the input.
        self.inputs = None
        self.aspects = None
        # sequence length
        self.sequence_length = None
        self.reverse_length = None
        self.aspect_length = None
        # the label input. (on-hot, the true label is 1.)
        self.lab_input = None
        self.embe_dict = None  # the embedding dictionary.
        # the optimize set.
        self.global_step = None  # the step counter.
        self.loss = None  # the loss of one batch evaluate.
        self.lr = None  # the learning rate.
        self.optimizer = None  # the optimiver.
        self.optimize = None  # the optimize action.
        # the mask of pre_train embedding.
        self.pe_mask = None
        # the predict.
        self.pred = None
        # the params need to be trained.
        self.params = None
        

    def GRU(self,inputs, layers,sequence_length,batch_size):
        cells = []
        for _ in range(layers):
            cell = rnn.GRUBlockCellV2(self.hidden_size*2)
            # cell = rnn.BasicLSTMCell(self.hidden_size*2)
            cells.append(cell)
        cell = rnn.MultiRNNCell(cells)
        zero_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=sequence_length, initial_state=zero_state)
        return outputs

        

    def build_model(self):
        '''
        build the MemNN model
        '''
        # the input.
        self.inputs = tf.placeholder(
            tf.int32,
            [None,None],
            name="inputs"
        )

        self.last_inputs = tf.placeholder(
            tf.int32,
            [None],
            name="last_inputs"
        )

        self.inputs_tags = tf.placeholder(
            tf.int32,
            [None,None],
            name="inputs_tags"
        )

        self.last_inputs_tags = tf.placeholder(
            tf.int32,
            [None],
            name="last_inputs_tags"
        )

        batch_size = tf.shape(self.inputs)[0]

        self.sequence_length = tf.placeholder(
            tf.int64,
            [None],
            name='sequence_length'
        )
        self.lab_input = tf.placeholder(
            tf.int32,
            [None],
            name="lab_input"
        )
        self.lab_input_tag = tf.placeholder(
            tf.int32,
            [None],
            name="lab_input_tag"
        )

        # self.lab_input = tf.concat((self.lab_input,self.lab_input_tag),axis=0)

        # the lookup dict.
        self.embe_dict = tf.Variable(
            self.pre_embedding,
            dtype=tf.float32,
            trainable=self.emb_up
        )
        self.pe_mask = tf.Variable(
            self.pre_embedding_mask,
            dtype=tf.float32,
            trainable=False
        )
        self.embe_dict *= self.pe_mask

        self.embe_dict_tag = tf.Variable(
            self.pre_embedding_tag,
            dtype=tf.float32,
            trainable=self.emb_up
        )
        self.pe_mask_tag = tf.Variable(
            self.pre_embedding_tag_mask,
            dtype=tf.float32,
            trainable=False
        )
        print_ops = []
        self.embe_dict_tag *= self.pe_mask_tag
        
        sent_bitmap = tf.ones_like(tf.cast(self.inputs, tf.float32))

        inputs_id = tf.nn.embedding_lookup(self.embe_dict, self.inputs,max_norm=1.5)
        lastinputs_id= tf.nn.embedding_lookup(self.embe_dict, self.last_inputs,max_norm=1.5)

        inputs_tag = tf.nn.embedding_lookup(self.embe_dict_tag,self.inputs_tags,max_norm=1.5)
        lastinputs_tag= tf.nn.embedding_lookup(self.embe_dict_tag, self.last_inputs_tags,max_norm=1.5)

        embe_dict_all_tag = tf.nn.embedding_lookup(self.embe_dict_tag,self.pre_tag,max_norm = 1.5)

        inputs = tf.concat((inputs_id,inputs_tag),axis=-1)
        lastinputs = tf.concat((lastinputs_id,lastinputs_tag),axis=-1)

        org_memory = inputs

        outputs = self.GRU(inputs,1,self.sequence_length,batch_size)

        # print_ops.append(tf.print('outputs.shape_before_reshape:       ',outputs.shape))
        outputs = outputs[:,-1,:]
        # outputs = tf.reshape(outputs,[-1,self.hidden_size*2])
        # print_ops.append(tf.print('outputs.shape_after_reshape:       ',outputs.shape))
        

        self.w1 = tf.Variable(
            tf.random_normal([self.edim, self.edim], stddev=self.stddev),
            trainable=True
        )

        prod = tf.tanh(tf.matmul(outputs,self.w1))
        self.embe_new_dict= tf.concat((self.embe_dict[1:],embe_dict_all_tag),axis=-1)

        sco_mat = tf.matmul(prod,self.embe_new_dict,transpose_b= True)
        with tf.control_dependencies(print_ops): 
            self.loss = legacy_seq2seq.sequence_loss_by_example([sco_mat], [self.lab_input], [tf.ones([tf.shape(sco_mat)[0]])])
            self.softmax_input = sco_mat
            # self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sco_mat,labels = self.lab_input)

            # the optimize.
            self.params = tf.trainable_variables()
            self.optimize = super(Seq2SeqAttNN, self).optimize_normal(
                self.loss, self.params)


    def train(self,sess, train_data, test_data=None,merged=None, writer=None,saver = None, threshold_acc=0.99):
        max_recall = []
        max_mrr = []
        max_train_acc = 0.0

        for i in range(len(self.cut_off)):
            max_recall.append(0.0)
            max_mrr.append(0.0)
        
        epoch_num = 0
        
        for epoch in range(self.nepoch):   # epoch round.
            if epoch>0:
                flag = True
            else:
                flag = False
            batch = 0
            c = []
            cost = 0.0  # the cost of each epoch.
            bt = batcher(
                samples=train_data.samples,
                class_num= self.n_items,
                random=True
            )
            while bt.has_next():    # batch round.
                # get this batch data
                batch_data = bt.next_batch()
                # build the feed_dict
                # for x,y in zip(batch_data['in_idxes'],batch_data['out_idxes']):
                batch_lenth = len(batch_data['in_idxes'])
                event = len(batch_data['in_idxes'][0])

                if batch_lenth > self.batch_size:
                    patch_len = int(batch_lenth / self.batch_size)
                    remain = int(batch_lenth % self.batch_size)
                    max_length = patch_len+1 if remain>0 else patch_len
                    i = 0
                    for x in range(patch_len):
                        tmp_in_data = batch_data['in_idxes'][i:i+self.batch_size]
                        tmp_out_data = batch_data['out_idxes'][i:i+self.batch_size]
                        tmp_in_tags = batch_data['in_tags'][i:i+self.batch_size]
                        tmp_out_tags = batch_data['out_tags'][i:i+self.batch_size]  

                        # for s in range(len(tmp_in_data[0])):
                        batch_in = []
                        batch_out = []
                        batch_last = []
                        batch_seq_l = []
                        for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                            # tmp_in = [i-1 for i in tmp_in]
                            # tmp_out = [i-1 for i in tmp_out]
                            _in = tmp_in[-1]
                            _out = int(np.mean(tmp_out))
                            batch_last.append(_in)
                            batch_in.append(tmp_in)
                            batch_out.append(_out)
                            batch_seq_l.append(len(tmp_in))

                        batch_in_tags = []
                        batch_out_tags = []
                        batch_last_tags = []
                        for tmp_in, tmp_out in zip(tmp_in_tags, tmp_out_tags):
                            # tmp_in = [i-1 for i in tmp_in]
                            # tmp_out = [i-1 for i in tmp_out]
                            _in = tmp_in[-1]
                            _out = int(np.mean(tmp_out))
                            batch_last_tags.append(_in)
                            batch_in_tags.append(tmp_in)
                            batch_out_tags.append(_out)
                        feed_dict = {
                            self.inputs: batch_in,
                            self.last_inputs: batch_last,
                            self.lab_input: batch_out,
                            self.sequence_length: batch_seq_l,
                            self.inputs_tags: batch_in_tags,
                            self.last_inputs_tags: batch_last_tags,
                            self.lab_input_tag: batch_out_tags

                        }
                        # train
                        crt_loss, crt_step, opt, embe_dict = sess.run(
                            [self.loss, self.global_step, self.optimize, self.embe_dict],
                            feed_dict=feed_dict
                        )
                        # graph = sess.run(merged,feed_dict=feed_dict)
                        # writer.add_summary(graph,batch+max_length*epoch)
                        c += list(crt_loss)
                        # print("Batch:" + str(batch) + ",cost:" + str(cost))
                        batch += 1
                        i += self.batch_size
                    if remain > 0:
                        # print (i, remain)
                        tmp_in_data = batch_data['in_idxes'][i:]
                        tmp_out_data = batch_data['out_idxes'][i:]
                        tmp_in_tags = batch_data['in_tags'][i:]
                        tmp_out_tags = batch_data['out_tags'][i:]
                        # for s in range(len(tmp_in_data[0])):
                        batch_in = []
                        batch_out = []
                        batch_last = []
                        batch_seq_l = []
                        for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                            # tmp_in = [i-1 for i in tmp_in]
                            # tmp_out = [i-1 for i in tmp_out]
                            _in = tmp_in[-1]
                            _out = int(np.mean(tmp_out))
                            batch_last.append(_in)
                            batch_in.append(tmp_in)
                            batch_out.append(_out)
                            batch_seq_l.append(len(tmp_in))
                        batch_in_tags = []
                        batch_out_tags = []
                        batch_last_tags = []
                        for tmp_in, tmp_out in zip(tmp_in_tags, tmp_out_tags):
                            # tmp_in = [i-1 for i in tmp_in]
                            # tmp_out = [i-1 for i in tmp_out]
                            _in = tmp_in[-1]
                            _out = int(np.mean(tmp_out))
                            batch_last_tags.append(_in)
                            batch_in_tags.append(tmp_in)
                            batch_out_tags.append(_out)

                        feed_dict = {
                            self.inputs: batch_in,
                            self.last_inputs: batch_last,
                            self.lab_input: batch_out,
                            self.sequence_length: batch_seq_l,
                            self.inputs_tags: batch_in_tags,
                            self.last_inputs_tags: batch_last_tags,
                            self.lab_input_tag: batch_out_tags

                        }
                        # train

                        crt_loss, crt_step, opt, embe_dict = sess.run(
                            [self.loss, self.global_step, self.optimize, self.embe_dict],
                            feed_dict=feed_dict
                        )
                        # graph = sess.run(merged,feed_dict=feed_dict)
                        # # tf.summary.scalar("train_loss",crt_loss)
                        # writer.add_summary(graph,batch+max_length*epoch)
                        # cost = np.mean(crt_loss)
                        c += list(crt_loss)
                        # print("Batch:" + str(batch) + ",cost:" + str(cost))
                        batch += 1
                else:
                    max_length = 1
                    tmp_in_data = batch_data['in_idxes']
                    tmp_out_data = batch_data['out_idxes']
                    tmp_in_tags = batch_data['in_tags'] #[i:i+self.batch_size]
                    tmp_out_tags = batch_data['out_tags']  #[i:i+self.batch_size]
                    # for s in range(len(tmp_in_data[0])):
                    batch_in = []
                    batch_out = []
                    batch_last = []
                    batch_seq_l = []
                    for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                        # tmp_in = [i-1 for i in tmp_in]
                        # tmp_out = [i-1 for i in tmp_out]
                        _in = tmp_in[-1]
                        _out = int(np.mean(tmp_out))
                        batch_last.append(_in)
                        batch_in.append(tmp_in)
                        batch_out.append(_out)
                        batch_seq_l.append(len(tmp_in))
                    batch_in_tags = []
                    batch_out_tags = []
                    batch_last_tags = []
                    for tmp_in, tmp_out in zip(tmp_in_tags, tmp_out_tags):
                        # tmp_in = [i-1 for i in tmp_in]
                        # tmp_out = [i-1 for i in tmp_out]
                        _in = tmp_in[-1]
                        _out = int(np.mean(tmp_out))
                        batch_last_tags.append(_in)
                        batch_in_tags.append(tmp_in)
                        batch_out_tags.append(_out)
                    feed_dict = {
                        self.inputs: batch_in,
                        self.last_inputs: batch_last,
                        self.lab_input: batch_out,
                        self.sequence_length: batch_seq_l,
                        self.inputs_tags: batch_in_tags,
                        self.last_inputs_tags: batch_last_tags,
                        self.lab_input_tag: batch_out_tags

                    }
                    # train
                    crt_loss, crt_step, opt, embe_dict = sess.run(
                        [self.loss, self.global_step, self.optimize, self.embe_dict],
                        feed_dict=feed_dict
                    )
                    # graph = sess.run(merged,feed_dict=feed_dict)
                    # # tf.summary.scalar("train_loss",crt_loss)
                    # writer.add_summary(graph,batch+max_length*epoch)

                    # cost = np.mean(crt_loss)
                    c+= list(crt_loss)
                    # print("Batch:" + str(batch) + ",cost:" + str(cost))
                    batch += 1
            # train_acc = self.test(sess,train_data)
            avgc = np.mean(c)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))
            if test_data != None:
                recall, mrr = self.test(sess, test_data)
                # tf.summary.scalar('recall',recall)
                # tf.summary.scalar('mrr',mrr)
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
                    if max_recall[i] > threshold_acc:
                        self.save_model(sess, self.config, saver)    
                    print ("max_recall@{}: ".format(str(self.cut_off[i])) + str(max_recall[i])+" max_mrr@{}: ".format(str(self.cut_off[i]))+str(max_mrr[i]))
                if increase_num==0:
                    epoch_num += 1
                if epoch_num==3:
                    print("长时间指标未增长，训练结束")
                    sys.exit(0)    
                test_data.flush()

        if self.is_print:
            TIPrint(test_data.samples, self.config,
                    {'recall': max_recall, 'mrr': max_mrr}, True)

    def test(self,sess,test_data):
        # calculate the acc
        print('Measuring Recall@{} and MRR@{}'.format(self.cut_off, self.cut_off))
        mrr, recall = [], []
        for i in range(len(self.cut_off)):
            mrr.append([])
            recall.append([])

        c_loss =[]
        batch = 0
        bt = batcher(
            samples = test_data.samples,
            class_num = self.n_items,
            random = False
        )
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("loss_log/", sess.graph)
        # summary_merge = tf.summary.merge_all() 
        while bt.has_next():    # batch round.
            # get this batch data
            batch_data = bt.next_batch()
            # build the feed_dict
            # for x,y in zip(batch_data['in_idxes'],batch_data['out_idxes']):
            batch_lenth = len(batch_data['in_idxes'])
            event = len(batch_data['in_idxes'][0])
            if batch_lenth > self.batch_size:
                patch_len = int(batch_lenth / self.batch_size)
                remain = int(batch_lenth % self.batch_size)
                i = 0
                for x in range(patch_len):
                    tmp_in_data = batch_data['in_idxes'][i:i+self.batch_size]
                    tmp_out_data = batch_data['out_idxes'][i:i+self.batch_size]
                    tmp_batch_ids = batch_data['batch_ids'][i:i+self.batch_size]

                    tmp_in_tags = batch_data['in_tags'][i:i+self.batch_size]
                    tmp_out_tags = batch_data['out_tags'][i:i+self.batch_size]

                    
                    # for s in range(len(tmp_in_data[0])):
                    batch_in = []
                    batch_out = []
                    batch_last = []
                    batch_seq_l = []
                    for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                        # tmp_in = [i-1 for i in tmp_in]
                        # tmp_out = [i-1 for i in tmp_out]
                        _in = tmp_in[-1]
                        _out = int(np.mean(tmp_out))
                        batch_last.append(_in)
                        batch_in.append(tmp_in)
                        batch_out.append(_out)
                        batch_seq_l.append(len(tmp_in))
                    
                    batch_in_tags = []
                    batch_out_tags = []
                    batch_last_tags = []
                    for tmp_in, tmp_out in zip(tmp_in_tags, tmp_out_tags):
                        # tmp_in = [i-1 for i in tmp_in]
                        # tmp_out = [i-1 for i in tmp_out]
                        _in = tmp_in[-1]
                        _out = int(np.mean(tmp_out))
                        batch_last_tags.append(_in)
                        batch_in_tags.append(tmp_in)
                        batch_out_tags.append(_out)


                    feed_dict = {
                        self.inputs: batch_in,
                        self.last_inputs: batch_last,
                        self.lab_input: batch_out,
                        self.sequence_length: batch_seq_l,
                        self.inputs_tags: batch_in_tags,
                        self.last_inputs_tags: batch_last_tags,
                        self.lab_input_tag: batch_out_tags
                    }
                    # train
                    preds, loss = sess.run(
                        [self.softmax_input, self.loss],
                        feed_dict=feed_dict
                    )
                    # graph = sess.run(merged,feed_dict=feed_dict)
                    # writer.add_summary(graph,batch)
                    t_r, t_m, ranks = cau_recall_mrr_org_list(preds, batch_out, cutoff=self.cut_off)
                    # test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
                    test_data.pack_preds(ranks, tmp_batch_ids)
                    # tf.summary.scalar("test_loss",loss)
                    # writer.add_summary(graph,batch)
                    c_loss += list(loss)
                    for k in range(len(self.cut_off)):
                        recall[k] += t_r[k]
                        mrr[k] += t_m[k]
                    batch += 1
                i += self.batch_size
                if remain > 0:
                    # print (i, remain)
                    tmp_in_data = batch_data['in_idxes'][i:]
                    tmp_out_data = batch_data['out_idxes'][i:]
                    tmp_batch_ids = batch_data['batch_ids'][i:]

                    tmp_in_tags = batch_data['in_tags'][i:]
                    tmp_out_tags = batch_data['out_tags'][i:]
                    # for s in range(len(tmp_in_data[0])):
                    batch_in = []
                    batch_out = []
                    batch_last = []
                    batch_seq_l = []
                    for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                        # tmp_in = [i-1 for i in tmp_in]
                        # tmp_out = [i-1 for i in tmp_out]
                        _in = tmp_in[-1]
                        _out = int(np.mean(tmp_out))
                        batch_last.append(_in)
                        batch_in.append(tmp_in)
                        batch_out.append(_out)
                        batch_seq_l.append(len(tmp_in))
                    batch_in_tags = []
                    batch_out_tags = []
                    batch_last_tags = []
                    for tmp_in, tmp_out in zip(tmp_in_tags, tmp_out_tags):
                        # tmp_in = [i-1 for i in tmp_in]
                        # tmp_out = [i-1 for i in tmp_out]
                        _in = tmp_in[-1]
                        _out = int(np.mean(tmp_out))
                        batch_last_tags.append(_in)
                        batch_in_tags.append(tmp_in)
                        batch_out_tags.append(_out)
                    feed_dict = {
                        self.inputs: batch_in,
                        self.last_inputs: batch_last,
                        self.lab_input: batch_out,
                        self.sequence_length: batch_seq_l,
                        self.inputs_tags: batch_in_tags,
                        self.last_inputs_tags: batch_last_tags,
                        self.lab_input_tag: batch_out_tags
                    }

                    # train
                    preds, loss = sess.run(
                        [self.softmax_input, self.loss],
                        feed_dict=feed_dict
                    )
                    # graph = sess.run(merged,feed_dict=feed_dict)
                    # writer.add_summary(graph,batch)
                    t_r, t_m, ranks = cau_recall_mrr_org_list(preds, batch_out, cutoff=self.cut_off)
                    # test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
                    test_data.pack_preds(ranks, tmp_batch_ids)
                    # tf.summary.scalar("test_loss",loss)
                    # writer.add_summary(graph,batch)
                    c_loss += list(loss)
                    for k in range(len(self.cut_off)):
                        recall[k] += t_r[k]
                        mrr[k] += t_m[k]
                    # recall += t_r
                    # mrr += t_m
                    batch += 1
            else:
                tmp_in_data = batch_data['in_idxes']
                tmp_out_data = batch_data['out_idxes']
                tmp_batch_ids = batch_data['batch_ids']
                tmp_in_tags = batch_data['in_tags'] #[i:i+self.batch_size]
                tmp_out_tags = batch_data['out_tags'] #[i:i+self.batch_size]
                # for s in range(len(tmp_in_data[0])):
                batch_in = []
                batch_out = []
                batch_last = []
                batch_seq_l = []
                for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                    # tmp_in = [i-1 for i in tmp_in]
                    # tmp_out = [i-1 for i in tmp_out]
                    _in = tmp_in[-1]
                    _out = int(np.mean(tmp_out))
                    batch_last.append(_in)
                    batch_in.append(tmp_in)
                    batch_out.append(_out)
                    batch_seq_l.append(len(tmp_in))
                batch_in_tags = []
                batch_out_tags = []
                batch_last_tags = []
                for tmp_in, tmp_out in zip(tmp_in_tags, tmp_out_tags):
                    # tmp_in = [i-1 for i in tmp_in]
                    # tmp_out = [i-1 for i in tmp_out]
                    _in = tmp_in[-1]
                    _out = int(np.mean(tmp_out))
                    batch_last_tags.append(_in)
                    batch_in_tags.append(tmp_in)
                    batch_out_tags.append(_out)
                feed_dict = {
                    self.inputs: batch_in,
                    self.last_inputs: batch_last,
                    self.lab_input: batch_out,
                    self.sequence_length: batch_seq_l,
                    self.inputs_tags: batch_in_tags,
                    self.last_inputs_tags: batch_last_tags,
                    self.lab_input_tag: batch_out_tags
                }

                # train
                preds, loss = sess.run(
                    [self.softmax_input, self.loss],
                    feed_dict=feed_dict
                )
                # graph = sess.run(merged,feed_dict=feed_dict)
                # writer.add_summary(graph,batch)
                t_r, t_m, ranks = cau_recall_mrr_org_list(preds, batch_out, cutoff=self.cut_off)
                # test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
                test_data.pack_preds(ranks, tmp_batch_ids)
                # tf.summary.scalar("test_loss",loss)
                # writer.add_summary(graph,batch)
                c_loss += list(loss)
                # recall += t_r
                # mrr += t_m
                for k in range(len(self.cut_off)):
                    recall[k] += t_r[k]
                    mrr[k] += t_m[k]
                batch += 1
        # for k in range(len(self.cut_off)):
        #     r, m =cau_samples_recall_mrr(test_data.samples,self.cut_off[k])
        #     print (r,m)
        # print (np.mean(c_loss))
        return  np.mean(recall,axis=1), np.mean(mrr,axis=1)
