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
            # self.max_num = config['max_num']  #video的最大编号

            # the pre-train embedding.
            ## shape = [nwords, edim]
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
                # self.pre_tag += self.item2tag[i]

            # a = [i for i in range((self.pre_embedding).shape[0])]
            # self.pre_tag = self.item2tag[a]

            # self.embe_dict_tag2id = []

            # self.pre_embedding_tag2id_mask = np.ones(np.shape(self.pre_embedding_tag2id))
            # self.pre_embedding_tag2id_mask[self.pad_idx] = 0

            # update the pre-train embedding or not.
            self.emb_up = config['emb_up']

            # the active function.
            self.active = config['active']

            # hidden size
            self.hidden_size = config['hidden_size']

            self.is_print = config['is_print']

            self.cut_off = config["cut_off"]
        
        self.batch_size_ = None
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
        
    def ln(self,inputs, epsilon = 1e-8, scope="ln"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            
            beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
        return outputs
    
    def simple_dnn(self,inputs, sub_name="mlp",hidden_units=[64, 32],single_output=False):       
        with tf.variable_scope("%s" % sub_name):
            x = inputs
            for i, units in enumerate(hidden_units):
                x = tf.layers.dense(x, units, activation=tf.nn.elu, name='layer_{}'.format(i), use_bias=False, kernel_initializer=tf.glorot_normal_initializer(), reuse=tf.AUTO_REUSE)
                x = self.ln(x, scope="ln_%d"%(i))
            if single_output:
                x = tf.layers.dense(x, 1, activation=None, name='linear', use_bias=False, kernel_initializer=tf.glorot_normal_initializer(), reuse=tf.AUTO_REUSE)
        return x 

    def attention_layer(self,recent_play, play_history, sub_name=None):
        recent_play = tf.tile(recent_play, [1, play_history.get_shape()[1], 1])
        din_input = tf.concat([recent_play, play_history, recent_play-play_history, recent_play*play_history], axis=-1)
        din_output = self.simple_dnn(din_input, hidden_units=[32], single_output=True, sub_name=sub_name+"_din")
        atten_score = tf.exp(din_output / (recent_play.get_shape().as_list()[-1] ** 0.5)) # scale
        atten_score /= (tf.reduce_sum(atten_score, axis=1, keepdims=True) + 1e-5) #防止除以0
        return tf.reduce_sum(atten_score*play_history, axis=1, keepdims=False)
    
    def target_attention(self,recent_play, play_history, seq_len,sub_name="target_attention", debug_name=""):
        with tf.variable_scope("%s" % sub_name, reuse=tf.AUTO_REUSE):
            # recent_play = tf.reshape(recent_play,[batch_size,1,recent_play.shape[1]])
            recent_play = tf.tile(recent_play, [1, 49, 1])
            play_history = play_history[:,:-1,...]
            din_input = tf.concat([recent_play, play_history, recent_play-play_history, recent_play*play_history], axis=-1) # [B, history_lens, 4*input_dims]
            din_output = self.simple_dnn(din_input, hidden_units=[32], single_output=True, sub_name="din")
            din_output = tf.nn.softmax(din_output, axis=1)
            return tf.reduce_sum(din_output*play_history, axis=1, keepdims=False)
    
    def get_norm_variable(self,var_name, input_dim, output_dim):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2.0 / (input_dim + output_dim)), seed=2023)
        return tf.get_variable(var_name, shape=(input_dim, output_dim), initializer=initializer)
    
    def unet(self,keys,  num_layers=3, out_dims=32, scope="unet"):
        """
        对输入的action_list长度敏感, 需要确认corp_size都为整数。
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            num_conv_per_layer = 1

            num_of_steps= 50
            # print("unet mask shape:", key_masks.get_shape())

            # compute padding & corp size
            corp_size = [num_of_steps + num_conv_per_layer*2]
            for _ in range(num_layers-1):
                corp_size.append(corp_size[-1]/2 + num_conv_per_layer*2)
                full_size = [corp_size[-1]]
            for _ in range(num_layers-1):
                full_size.insert(0, full_size[0]*2 + num_conv_per_layer*2)
                filters = [out_dims]
            for _ in range(num_layers-1):
                filters.append(filters[-1]*2)

            #前后拼接一部分镜像序列
            padding_size = int((full_size[0] - num_of_steps) / 2)
            inputs = tf.concat([tf.reverse(keys[:, :padding_size, :], axis=[1]),keys, tf.reverse(keys[:, -padding_size:, :], axis=[1])], axis=1)
            # top down
            left_blocks = []
            x = inputs
            for i in range(num_layers):
                for j in range(num_conv_per_layer):
                    x = tf.layers.conv1d(x, filters=filters[i], kernel_size=3, strides=1, padding='valid', activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2())
                    x = self.ln(x, scope="down_ln_%d_%d"%(i, j))
                left_blocks.append(x)
                if i != num_layers-1:
                    x = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='valid') # 下采样
            print('bottom tmp shape:%s'%x.get_shape())

            for i in range(num_layers-2, -1, -1):
                edge_size = int((full_size[i] - num_conv_per_layer*2 - corp_size[i]) / 2)
                x = tf.keras.layers.UpSampling1D(size=2)(x) # 上采样
                x = tf.layers.conv1d(x, filters=filters[i], kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2())
                x = tf.concat([x, left_blocks[i][:, edge_size: -edge_size, :]], axis=-1) # 拼接裁剪后的低维信息
                for j in range(num_conv_per_layer):
                    x = tf.layers.conv1d(x, filters=filters[i], kernel_size=3, strides=1, padding='valid', activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2())
                    x = self.ln(x, scope="up_ln_%d_%d"%(i, j))
                
            # key_masks = tf.tile(key_masks, [1, 1, tf.shape(x)[-1]])
            # outputs = x * key_masks
            outputs = x
            print('unet output shape:%s'%outputs.get_shape())
            return outputs

    def get_actual_seq_len(self,feature_seqs):
        used = tf.sign(tf.reduce_max(tf.abs(feature_seqs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        return length 
        
    def get_mask(self,play_list):
        actual_history_action_num = self.get_actual_seq_len(play_list)     
        key_masks = tf.sequence_mask(actual_history_action_num, tf.shape(play_list)[1])   # [B, T]
        key_masks = tf.cast(tf.expand_dims(key_masks, axis=2), tf.float32)
        return key_masks
    
    def self_attention(self,inputs, inputs_action_len, atten_unit=16, head_num=4, scope_name="self_attention", debug_name=""):
        """
        inputs: user action list, (batch_size x max_action_length x action_dim)
        return concat(inputs, context_info)
        """
        action_dim = inputs.get_shape().as_list()[-1]
        # max_action_len = inputs.get_shape().as_list()[1]
        max_action_len = 50
        with tf.variable_scope(scope_name):
            mask = tf.cast(tf.sequence_mask(tf.reshape(inputs_action_len, [-1]), inputs.get_shape().as_list()[1]), tf.float32) #(batch_size, max_action_length)
            # print(mask.shape)
            Q = self.get_norm_variable("Q_mat", action_dim, atten_unit * head_num) #(action_dim, atten_unit * head_num)
            K = self.get_norm_variable("K_mat", action_dim, atten_unit * head_num) #(action_dim, atten_unit * head_num)
            V = self.get_norm_variable("V_mat", action_dim, atten_unit * head_num) #(action_dim, atten_unit * head_num)
            q = tf.tensordot(inputs, Q, axes=(-1,0)) #(batch_size, max_action_length, atten_unit * head_num)
            k = tf.tensordot(inputs, K, axes=(-1,0)) #(batch_size, max_action_length, atten_unit * head_num)
            v = tf.tensordot(inputs, V, axes=(-1,0)) #(batch_size, max_action_length, atten_unit * head_num)
            q = tf.concat(tf.split(q, head_num, axis=2), axis=0) #(head_num*batch_size, max_action_length, atten_unit)
            k = tf.concat(tf.split(k, head_num, axis=2), axis=0) #(head_num*batch_size, max_action_length, atten_unit)
            v = tf.concat(tf.split(v, head_num, axis=2), axis=0) #(head_num*batch_size, max_action_length, atten_unit)
            inner_product = tf.matmul(q, k, transpose_b=True) / np.sqrt(atten_unit) #(head_num*batch_size, max_action_length, max_action_length)
            mask = tf.tile(mask, [head_num, 1]) #(head_num * batch_size, max_action_length)
            # print(mask.shape)
            mask = tf.tile(tf.expand_dims(mask, 1), [1, max_action_len, 1]) #(head_num * batch_size, max_action_length, max_action_length)
            padding = tf.ones_like(inner_product) * (-2 ** 32 + 1)
            inner_product = tf.matrix_set_diag(inner_product, tf.ones_like(inner_product)[:,:,0] * (-2 ** 32 + 1))
            # print(mask.shape)

            # print(inner_product.shape)
            
            # print(padding.shape)
            inner_product = tf.where(tf.equal(mask, 1), inner_product, padding)
            inner_product = tf.nn.softmax(inner_product) #(head_num * batch_size, max_action_length, max_action_length)
            
            outputs = tf.matmul(inner_product, v) #(head_num*batch_size, max_action_length, atten_unit)
            outputs = tf.concat(tf.split(outputs, head_num, axis=0), axis=2) #(batch_size, max_action_length, atten_unit * head_num)
            outputs = tf.layers.dense(outputs, action_dim, name='linear', use_bias=False, kernel_initializer=tf.glorot_normal_initializer(), reuse=tf.AUTO_REUSE) # linear
            outputs += inputs # add
            outputs = self.ln(outputs, scope="ln_1") # norm
            ffn_outputs = self.simple_dnn(outputs, hidden_units=[64, action_dim]) # ffn
            outputs += ffn_outputs # add
            outputs = self.ln(outputs, scope="ln_2") # norm
            return outputs

    def build_model(self):
        '''
        build the MemNN model
        '''
        # the input.
        self.inputs = tf.placeholder(
            tf.int32,
            [None,50],
            name="inputs"
        )

        self.last_inputs = tf.placeholder(
            tf.int32,
            [None],
            name="last_inputs"
        )

        self.inputs_tags = tf.placeholder(
            tf.float32,
            [None,50,None],
            name="inputs_tags"
        )

        self.last_inputs_tags = tf.placeholder(
            tf.float32,
            [None,None],
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
            tf.float32,
            [None,None],
            name="lab_input_tag"
        )

        self.bs = tf.placeholder(tf.int32)
        # self.lab_input = tf.concat((self.lab_input,self.lab_input_tag),axis=0)
        # self.batch_size_ = tf.placeholder(
        #     tf.int64,
        #     None,
        #     name='batch_size'
        # )
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
        self.embe_dict_tag *= self.pe_mask_tag


        inputs_id = tf.nn.embedding_lookup(self.embe_dict, self.inputs,max_norm=1.5)
        lastinputs_id= tf.nn.embedding_lookup(self.embe_dict, self.last_inputs,max_norm=1.5)

        inputs_tag = tf.reshape(self.inputs_tags,[-1,tf.shape(self.inputs_tags)[-1]])
        inputs_tag = tf.matmul(inputs_tag,self.embe_dict_tag)
        inputs_tag = tf.reshape(inputs_tag,[tf.shape(self.inputs_tags)[0],tf.shape(self.inputs_tags)[1],2])/tf.reduce_sum(self.inputs_tags,axis=-1,keepdims=True) 
        # inputs_tag = tf.math.l2_normalize(inputs_tag)
        lastinputs_tag = tf.matmul(self.last_inputs_tags,self.embe_dict_tag)/tf.reduce_sum(self.last_inputs_tags,axis=-1,keepdims=True)  
        # inputs_tag = tf.math.l2_normalize(lastinputs_tag)
        self.pre_tag = tf.cast(self.pre_tag,tf.float32)
        embe_dict_all_tag = tf.matmul(self.pre_tag[1:],self.embe_dict_tag)/tf.reduce_sum(self.pre_tag[1:],axis=-1,keepdims=True)


        inputs = tf.concat((inputs_id,inputs_tag),axis=-1)
        lastinputs = tf.concat((lastinputs_id,lastinputs_tag),axis=-1)

        sent_bitmap = tf.ones_like(tf.cast(self.inputs, tf.float32))

        org_memory = inputs

        lastinputs = tf.expand_dims(lastinputs,1)


        history_play_actual_lens = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(inputs), axis=2, keepdims=True)), axis=1, keepdims=True), tf.float32)
        
        
        # inputs_context = self.self_attention(inputs,history_play_actual_lens)

        # inputs_mask = self.get_mask(inputs)
        inputs_context = self.unet(inputs,num_layers=3, out_dims=4)
        

        inputs = tf.concat([inputs, inputs_context], axis=-1)
        # # print(inputs.shape)
        inputs = self.simple_dnn(inputs, sub_name="long_mlp", hidden_units=[4]) 
        # print(inputs.shape)

        din_out  = self.target_attention(lastinputs,inputs,self.sequence_length)
        
        lastinputs = tf.squeeze(lastinputs,axis=1)

        din_out = tf.concat((din_out,lastinputs),axis = -1)

        self.w1 = tf.Variable(
            tf.random_normal([self.edim*2, self.edim], stddev=self.stddev),
            trainable=True
        )

        din_out = tf.tanh(tf.matmul(din_out,self.w1))

        self.embe_new_dict= tf.concat((self.embe_dict[2:],embe_dict_all_tag),axis=-1)

        sco_mat = tf.matmul(din_out,self.embe_new_dict,transpose_b= True)
        
        # self.lab_input_id = tf.cast(self.lab_input_id,tf.float32)
        
        with tf.name_scope('loss'):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sco_mat,labels = self.lab_input)
            loss = tf.reduce_mean(self.loss)
            tf.summary.scalar("loss",loss)

        sco_mat = tf.sigmoid(sco_mat)
        self.softmax_input = sco_mat
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
                            _out = tmp_out[0]
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
                            _out = tmp_out[0]
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
                            self.lab_input_tag: batch_out_tags,
                            self.bs : len(batch_in)

                        }
                        self.batch_size_ = len(batch_in)
                        # train
                        crt_loss, crt_step, opt, embe_dict = sess.run(
                            [self.loss, self.global_step, self.optimize, self.embe_dict],
                            feed_dict=feed_dict
                        )
                        graph = sess.run(merged,feed_dict=feed_dict)
                        writer.add_summary(graph,batch+max_length*epoch)
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
                            _out = tmp_out[0]
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
                            _out = tmp_out[0]
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
                            self.lab_input_tag: batch_out_tags,
                            self.bs : len(batch_in)
                        }
                        # train
                        self.batch_size_ = len(batch_in)
                        crt_loss, crt_step, opt, embe_dict = sess.run(
                            [self.loss, self.global_step, self.optimize, self.embe_dict],
                            feed_dict=feed_dict
                        )
                        graph = sess.run(merged,feed_dict=feed_dict)
                        # tf.summary.scalar("train_loss",crt_loss)
                        writer.add_summary(graph,batch+max_length*epoch)
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
                        _out = tmp_out[0]
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
                        _out = tmp_out[0]
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
                        self.lab_input_tag: batch_out_tags,
                        self.bs : len(batch_in)
                    }
                    self.batch_size_ = len(batch_in)
                    # train
                    crt_loss, crt_step, opt, embe_dict = sess.run(
                        [self.loss, self.global_step, self.optimize, self.embe_dict],
                        feed_dict=feed_dict
                    )
                    graph = sess.run(merged,feed_dict=feed_dict)
                    # tf.summary.scalar("train_loss",crt_loss)
                    writer.add_summary(graph,batch+max_length*epoch)

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
                        _out = tmp_out[0]
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
                        _out = tmp_out[0]
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
                        self.lab_input_tag: batch_out_tags,
                        self.bs : len(batch_in)
                    }
                    self.batch_size_ = len(batch_in)
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
                        _out = tmp_out[0]
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
                        _out = tmp_out[0]
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
                        self.lab_input_tag: batch_out_tags,
                        self.bs : len(batch_in)
                    }
                    self.batch_size_ = len(batch_in)
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
                    _out = tmp_out[0]
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
                    _out = tmp_out[0]
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
                    self.lab_input_tag: batch_out_tags,
                    self.bs : len(batch_in)
                }
                self.batch_size_ = len(batch_in)
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

