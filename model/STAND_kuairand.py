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
import math

class T2diff(NN):
    """
    The memory network with context attention.
    """
    def __init__(self, config):
        super(T2diff, self).__init__(config)
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
        self.train_flag = True
        
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
        with tf.variable_scope("%s" % sub_name, reuse=tf.AUTO_REUSE):
            x = inputs
            for i, units in enumerate(hidden_units):
                x = tf.layers.dense(x, units, activation=tf.nn.elu, name='layer_{}'.format(i), use_bias=False, kernel_initializer=tf.glorot_normal_initializer(), reuse=tf.AUTO_REUSE)
                x = self.ln(x, scope="ln_%d"%(i))
            if single_output:
                x = tf.layers.dense(x, 1, activation=None, name='linear', use_bias=False, kernel_initializer=tf.glorot_normal_initializer(), reuse=tf.AUTO_REUSE)
        return x 
    
    def target_attention(self,recent_play, play_history,sub_name="target_attention"):
        with tf.variable_scope("%s" % sub_name, reuse=tf.AUTO_REUSE):
            recent_play = tf.tile(recent_play, [1, 40, 1])
            din_input = tf.concat([recent_play, play_history, recent_play-play_history, recent_play*play_history], axis=-1)
            din_output = self.simple_dnn(din_input, hidden_units=[self.edim], single_output=True, sub_name="din")
            din_output = tf.nn.softmax(din_output, axis=1)
            return tf.reduce_sum(din_output*play_history, axis=1, keepdims=False)
    
    def get_norm_variable(self,var_name, input_dim, output_dim):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2.0 / (input_dim + output_dim)), seed=2023)
        return tf.get_variable(var_name, shape=(input_dim, output_dim), initializer=initializer)

    def unet(self,keys,  num_layers=3, out_dims=4, scope="unet"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            num_conv_per_layer = 1
            num_of_steps= 50
            corp_size = [num_of_steps + num_conv_per_layer*2]
            for _ in range(num_layers-1):
                corp_size.append(corp_size[-1]/2 + num_conv_per_layer*2)
                full_size = [corp_size[-1]]
            for _ in range(num_layers-1):
                full_size.insert(0, full_size[0]*2 + num_conv_per_layer*2)
                filters = [out_dims]
            for _ in range(num_layers-1):
                filters.append(filters[-1]*2)
            padding_size = int((full_size[0] - num_of_steps) / 2)
            inputs = tf.concat([tf.reverse(keys[:, :padding_size, :], axis=[1]),keys, tf.reverse(keys[:, -padding_size:, :], axis=[1])], axis=1)
            left_blocks = []
            x = inputs
            for i in range(num_layers):
                for j in range(num_conv_per_layer):
                    x = tf.layers.conv1d(x, filters=filters[i], kernel_size=3, strides=1, padding='valid', activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2())
                    x = tf.nn.swish(x)
                    
                    x = self.ln(x, scope="down_ln_%d_%d"%(i, j))
                left_blocks.append(x)
                if i != num_layers-1:
                    x = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='valid') 

            for i in range(num_layers-2, -1, -1):
                edge_size = int((full_size[i] - num_conv_per_layer*2 - corp_size[i]) / 2)
                x = tf.keras.layers.UpSampling1D(size=2)(x) # 
                x = tf.layers.conv1d(x, filters=filters[i], kernel_size=1, strides=1, padding='valid', activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2())
                x = tf.nn.swish(x)
                x = tf.concat([x, left_blocks[i][:, edge_size: -edge_size, :]], axis=-1) 
                for j in range(num_conv_per_layer):
                    x = tf.layers.conv1d(x, filters=filters[i], kernel_size=3, strides=1, padding='valid', activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2())
                    x = tf.nn.swish(x)
                    x = self.ln(x, scope="up_ln_%d_%d"%(i, j))
            outputs = x

            return outputs
    
    def transformer(self,inputs, inputs_action_len, atten_unit=16, head_num=4, scope_name="transformer", debug_name=""):
        """
        inputs: user action list, (batch_size x max_action_length x action_dim)
        return concat(inputs, context_info)
        """
        action_dim = inputs.get_shape().as_list()[-1]
        with tf.variable_scope(scope_name, reuse= tf.AUTO_REUSE):
            Q = self.get_norm_variable("Q_mat", action_dim, atten_unit * head_num) 
            K = self.get_norm_variable("K_mat", action_dim, atten_unit * head_num) 
            V = self.get_norm_variable("V_mat", action_dim, atten_unit * head_num) 
            q = tf.tensordot(inputs, Q, axes=(-1,0)) 
            k = tf.tensordot(inputs, K, axes=(-1,0)) 
            v = tf.tensordot(inputs, V, axes=(-1,0)) 
            q = tf.concat(tf.split(q, head_num, axis=2), axis=0) 
            k = tf.concat(tf.split(k, head_num, axis=2), axis=0) 
            v = tf.concat(tf.split(v, head_num, axis=2), axis=0) 
            inner_product = tf.matmul(q, k, transpose_b=True) / np.sqrt(atten_unit) 
            inner_product = tf.matrix_set_diag(inner_product, tf.ones_like(inner_product)[:,:,0] * (-2 ** 32 + 1))
            inner_product = tf.nn.softmax(inner_product) 
            
            outputs = tf.matmul(inner_product, v) 
            outputs = tf.concat(tf.split(outputs, head_num, axis=0), axis=2) 
            outputs = tf.layers.dense(outputs, action_dim, name='linear', use_bias=False, kernel_initializer=tf.glorot_normal_initializer(), reuse=tf.AUTO_REUSE) # linear
            outputs += inputs
            outputs = self.ln(outputs, scope="ln_1") 
            ffn_outputs = self.simple_dnn(outputs, hidden_units=[action_dim]) 
            outputs += ffn_outputs 
            outputs = self.ln(outputs, scope="ln_2")
            return outputs
    
    def diffusion(self, inputs, input_with_target, is_train, sub_name="diffusion", total_steps=5):
        with tf.variable_scope("%s" % sub_name):
            action_lens = 50
            hidden_dim = inputs.get_shape().as_list()[-1]
            inputs = tf.stop_gradient(inputs)
            input_with_target = tf.stop_gradient(input_with_target)
            a, b = 1e-2, 4.5
            alphas, betas, alpha_hats, beta_hats = [], [], [], [] 
            for s in range(1, total_steps+1):
                beta = a*math.exp(b*s/total_steps) # exponential schedule
                betas.append(beta)
                alphas.append(1-beta)
                alpha_hats.append(1-beta) if s == 1 else alpha_hats.append((1-beta)*alpha_hats[-1])
                beta_hats.append(0) if s == 1 else beta_hats.append((1-alpha_hats[-2])/(1-alpha_hats[-1])*beta)

            step_weight = 1e-2
            step_embs = [[math.sin(t/2**f) for f in range(hidden_dim*2)] for t in range(1, total_steps+1)] 
            
            if is_train==1: # training process
                t = tf.random.uniform(shape=[tf.shape(inputs)[0]], minval=0, maxval=total_steps, dtype=tf.int32)
                diff = input_with_target - inputs

                gaussian_noise = tf.random.normal(shape=tf.shape(diff), mean=0, stddev=1)
                alpha_hats = tf.convert_to_tensor(alpha_hats, dtype=tf.float32)
                alpha_hats = tf.gather(alpha_hats, t, axis=0)
                alpha_hats = tf.expand_dims(tf.expand_dims(alpha_hats, axis=1), axis=2)
                alpha_hats = tf.tile(alpha_hats, [1, action_lens, hidden_dim])
                noised_t_input = tf.sqrt(alpha_hats)*diff + tf.sqrt(1-alpha_hats)*gaussian_noise
                noised_t_input = tf.concat((noised_t_input, inputs),axis=-1)

                step_emb = tf.convert_to_tensor(step_embs, dtype=tf.float32)
                step_emb = tf.gather(step_emb, t, axis=0)
                step_emb = tf.expand_dims(step_emb, axis=1)
                step_emb = tf.tile(step_emb, [1, action_lens, 1])
                reconstructed_t = self.unet(noised_t_input + step_weight*step_emb, num_layers=4, out_dims=hidden_dim) 

                KL_loss = tf.math.reduce_mean(tf.sqrt(tf.math.reduce_sum(tf.square(diff-reconstructed_t), axis=-1))) 

                predicted_next = (inputs + reconstructed_t)[:, -1:, :] 
            else: # inference process
                KL_loss = None
                infer_steps = total_steps
                gaussian_noise_diffu = tf.random.normal(shape=tf.shape(inputs), mean=0, stddev=1)
                noised_t = gaussian_noise_diffu
                for t in range(infer_steps-1, -1, -1): 
                    noised_t_input = tf.concat((noised_t,inputs),axis = -1)                
                    
                    step_emb = tf.convert_to_tensor(step_embs[t], dtype=tf.float32)
                    reconstructed_t = self.unet(noised_t_input + step_weight*step_emb, num_layers=4, out_dims=hidden_dim)
                    if t == 0:
                        noised_t = reconstructed_t
                    else:
                        gaussian_noise_reverse = tf.random.normal(shape=tf.shape(inputs), mean=0, stddev=1) 
                        noised_t = tf.sqrt(alpha_hats[t-1])*betas[t]/(1-alpha_hats[t])*reconstructed_t + tf.sqrt(alpha_hats[t])*(1-alpha_hats[t-1])/(1-alpha_hats[t])*noised_t + beta_hats[t]*gaussian_noise_reverse 
                
                predicted_next = (inputs + noised_t)[:, -1:, :] # [B, h]
        return KL_loss, predicted_next

    def build_train_model(self):
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
        with tf.variable_scope('model'):
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

        print_ops = []
        
        inputs_id = tf.nn.embedding_lookup(self.embe_dict, self.inputs,max_norm=1.5)
        lab_input_emb = tf.nn.embedding_lookup(self.embe_dict,self.lab_input,max_norm=1.5)

        inputs_tag = tf.nn.embedding_lookup(self.embe_dict_tag,self.inputs_tags,max_norm=1.5)
        lab_input_tag_emb = tf.nn.embedding_lookup(self.embe_dict_tag,self.lab_input_tag,max_norm=1.5)

        embe_dict_all_tag = tf.nn.embedding_lookup(self.embe_dict_tag,self.pre_tag,max_norm = 1.5)

        inputs = tf.concat((inputs_id,inputs_tag),axis=-1)
        lab_input_emb = tf.concat((lab_input_emb,lab_input_tag_emb),axis=-1)
        lab_input_emb = tf.expand_dims(lab_input_emb,1)

        extended_input = tf.concat([inputs, lab_input_emb], axis=1)
        KL_loss, predicted_next = self.diffusion(extended_input[:, : -1, :], extended_input[:, 1: , :], 1)
        
        session_inputs = tf.concat([extended_input[:, -11:-1, :], predicted_next], axis=1)
        history_play_actual_lens = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(session_inputs), axis=2, keepdims=True)), axis=1, keepdims=True), tf.float32) # zero-mask
        session_outputs = self.transformer(session_inputs, history_play_actual_lens, atten_unit=self.edim//2, head_num=2)
        session_outputs = tf.math.reduce_mean(session_outputs, 1, True)
        din_out  = self.target_attention(session_outputs, extended_input[:, :-11, ])
        
        session_outputs = tf.squeeze(session_outputs,axis=1)
        user_emb = tf.concat((din_out,session_outputs),axis = -1)

        self.embe_new_dict= tf.concat((self.embe_dict[1:],embe_dict_all_tag),axis=-1)
        self.embe_new_dict = self.simple_dnn(self.embe_new_dict, sub_name="mlp", hidden_units=[2*self.edim]) 
        sco_mat = tf.matmul(user_emb,self.embe_new_dict,transpose_b= True)
        
        with tf.control_dependencies(print_ops):   
            with tf.name_scope('train_loss'):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sco_mat,labels = self.lab_input)
                loss = tf.reduce_mean(self.loss)
                tf.summary.scalar("loss",loss)
                
                self.loss_kl = KL_loss
                if KL_loss is not None:
                    tf.summary.scalar("KL_loss",KL_loss)
                    self.loss = self.loss + 10*self.loss_kl
            self.softmax_input = sco_mat

            self.params = tf.trainable_variables()
            self.optimize = self.optimize_normal(
                self.loss, self.params)
    
    def build_test_model(self):
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

        with tf.variable_scope('model'):
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

        print_ops = []
        
        inputs_id = tf.nn.embedding_lookup(self.embe_dict, self.inputs,max_norm=1.5)
        lab_input_emb = tf.nn.embedding_lookup(self.embe_dict,self.lab_input,max_norm=1.5)

        inputs_tag = tf.nn.embedding_lookup(self.embe_dict_tag,self.inputs_tags,max_norm=1.5)
        lab_input_tag_emb = tf.nn.embedding_lookup(self.embe_dict_tag,self.lab_input_tag,max_norm=1.5)

        embe_dict_all_tag = tf.nn.embedding_lookup(self.embe_dict_tag,self.pre_tag,max_norm = 1.5)

        inputs = tf.concat((inputs_id,inputs_tag),axis=-1)
        lab_input_emb = tf.concat((lab_input_emb,lab_input_tag_emb),axis=-1)
        lab_input_emb = tf.expand_dims(lab_input_emb,1)


        extended_input = tf.concat([inputs, lab_input_emb], axis=1)
        KL_loss, predicted_next = self.diffusion(extended_input[:, : -1, :], extended_input[:, 1: , :], 0)
        session_inputs = tf.concat([extended_input[:, -11: -1, :], predicted_next], axis=1)
        history_play_actual_lens = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(session_inputs), axis=2, keepdims=True)), axis=1, keepdims=True), tf.float32)
        session_outputs = self.transformer(session_inputs, history_play_actual_lens, atten_unit=self.edim//2, head_num=2)
        session_outputs = tf.math.reduce_mean(session_outputs, 1, True)


        din_out  = self.target_attention(session_outputs, extended_input[:, :-11, ])
        
        session_outputs = tf.squeeze(session_outputs,axis=1)

        user_emb = tf.concat((din_out,session_outputs),axis = -1)

        self.embe_new_dict= tf.concat((self.embe_dict[1:],embe_dict_all_tag),axis=-1)
        self.embe_new_dict = self.simple_dnn(self.embe_new_dict, sub_name="mlp", hidden_units=[2*self.edim]) 
        sco_mat = tf.matmul(user_emb,self.embe_new_dict,transpose_b= True)

        with tf.name_scope('test_loss'):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sco_mat,labels = self.lab_input)
            loss = tf.reduce_mean(self.loss)
            tf.summary.scalar("loss",loss)
        self.softmax_input = sco_mat
    
    def train(self,sess, epoch, train_data, merged=None, writer=None, threshold_acc=0.99):   
        batch = 0
        c = []
        c_kl = []
        cost = 0.0  
        bt = batcher(
            samples=train_data.samples,
            class_num= self.n_items,
            random=True
        )
        while bt.has_next():   
            batch_data = bt.next_batch()
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

                    batch_in = []
                    batch_out = []
                    batch_last = []
                    batch_seq_l = []
                    for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
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
                    self.batch_size_ = len(batch_in)
                    crt_loss, crt_step, opt = sess.run(
                        [self.loss, self.global_step, self.optimize],
                        feed_dict=feed_dict
                    )
                    graph = sess.run(merged,feed_dict=feed_dict)
                    writer.add_summary(graph,batch+max_length*epoch)
                    c += list(crt_loss)

                    batch += 1
                    i += self.batch_size
                if remain > 0:
                    tmp_in_data = batch_data['in_idxes'][i:]
                    tmp_out_data = batch_data['out_idxes'][i:]
                    tmp_in_tags = batch_data['in_tags'][i:]
                    tmp_out_tags = batch_data['out_tags'][i:]
                    batch_in = []
                    batch_out = []
                    batch_last = []
                    batch_seq_l = []
                    for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
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
                    
                    self.batch_size_ = len(batch_in)
                    crt_loss, crt_step, opt = sess.run(
                        [self.loss, self.global_step, self.optimize],
                        feed_dict=feed_dict
                    )
                    graph = sess.run(merged,feed_dict=feed_dict)
                    writer.add_summary(graph,batch+max_length*epoch)
                    c += list(crt_loss)
                    batch += 1
            else:
                max_length = 1
                tmp_in_data = batch_data['in_idxes']
                tmp_out_data = batch_data['out_idxes']
                tmp_in_tags = batch_data['in_tags'] 
                tmp_out_tags = batch_data['out_tags']  
                batch_in = []
                batch_out = []
                batch_last = []
                batch_seq_l = []
                for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
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
                self.batch_size_ = len(batch_in)

                crt_loss, crt_step, opt = sess.run(
                    [self.loss, self.global_step, self.optimize],
                    feed_dict=feed_dict
                )
                graph = sess.run(merged,feed_dict=feed_dict)
                writer.add_summary(graph,batch+max_length*epoch)

                c += list(crt_loss)
                batch += 1
        avgc = np.mean(c)
        if np.isnan(avgc):
            print('Epoch {}: NaN error!'.format(str(epoch)))
            self.error_during_train = True
            return
        print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))
            

    def test(self,sess,test_data):
        self.train_flag = False
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
        while bt.has_next():   
            batch_data = bt.next_batch()
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

                    batch_in = []
                    batch_out = []
                    batch_last = []
                    batch_seq_l = []
                    for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
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
                        self.lab_input_tag: batch_out_tags
                    }
                    self.batch_size_ = len(batch_in)
                    preds, loss = sess.run(
                        [self.softmax_input, self.loss],
                        feed_dict=feed_dict
                    )
                    t_r, t_m, ranks = cau_recall_mrr_org_list(preds, batch_out, cutoff=self.cut_off)
                    test_data.pack_preds(ranks, tmp_batch_ids)
                    c_loss += list(loss)
                    for k in range(len(self.cut_off)):
                        recall[k] += t_r[k]
                        mrr[k] += t_m[k]
                    batch += 1
                i += self.batch_size
                if remain > 0:
                    tmp_in_data = batch_data['in_idxes'][i:]
                    tmp_out_data = batch_data['out_idxes'][i:]
                    tmp_batch_ids = batch_data['batch_ids'][i:]

                    tmp_in_tags = batch_data['in_tags'][i:]
                    tmp_out_tags = batch_data['out_tags'][i:]
                    batch_in = []
                    batch_out = []
                    batch_last = []
                    batch_seq_l = []
                    for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
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
                        self.lab_input_tag: batch_out_tags
                    }
                    self.batch_size_ = len(batch_in)
                    preds, loss = sess.run(
                        [self.softmax_input, self.loss],
                        feed_dict=feed_dict
                    )
                    t_r, t_m, ranks = cau_recall_mrr_org_list(preds, batch_out, cutoff=self.cut_off)
                    test_data.pack_preds(ranks, tmp_batch_ids)
                    c_loss += list(loss)
                    for k in range(len(self.cut_off)):
                        recall[k] += t_r[k]
                        mrr[k] += t_m[k]
                    batch += 1
            else:
                tmp_in_data = batch_data['in_idxes']
                tmp_out_data = batch_data['out_idxes']
                tmp_batch_ids = batch_data['batch_ids']
                tmp_in_tags = batch_data['in_tags'] 
                tmp_out_tags = batch_data['out_tags']
                batch_in = []
                batch_out = []
                batch_last = []
                batch_seq_l = []
                for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
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
                    self.lab_input_tag: batch_out_tags
                }
                self.batch_size_ = len(batch_in)
                preds, loss = sess.run(
                    [self.softmax_input, self.loss],
                    feed_dict=feed_dict
                )
                t_r, t_m, ranks = cau_recall_mrr_org_list(preds, batch_out, cutoff=self.cut_off)
                test_data.pack_preds(ranks, tmp_batch_ids)
                c_loss += list(loss)
                for k in range(len(self.cut_off)):
                    recall[k] += t_r[k]
                    mrr[k] += t_m[k]
                batch += 1
        return  np.mean(recall,axis=1), np.mean(mrr,axis=1)