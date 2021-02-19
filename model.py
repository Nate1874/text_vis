import tensorflow as tf
import numpy as np
import random
import copy
from data_reader import data_reader 
from helper import helper
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import operator
import scipy.io as sio
import time



class CNN(object):

    def __init__(self, sess, flag):
        self.conf = flag
        self.sess = sess
        self.initializer = tf.contrib.layers.xavier_initializer()
        if self.conf.isTraining == True:
            self.input_x = tf.placeholder(tf.int32, [None, self.conf.sequence_length], name="input_x")
        else:
            self.initialized_x = tf.get_variable("optimized_input", shape=[1, self.conf.sequence_length, self.conf.embed_size],initializer=self.initializer)
            self.input_x = tf.placeholder(tf.int32, [None, self.conf.sequence_length], name="input_x")
            self.input_x2 = tf.placeholder(tf.int32, [None, self.conf.sequence_length], name="input_x2")
        self.input_mask  = tf.placeholder(tf.int32, [None, self.conf.sequence_length], name= "input_mask")
        self.input_y = tf.placeholder(tf.int32, [None, self.conf.num_classes], name="input_y")
        self.label = tf.placeholder(tf.int32, [None], name="label")
        self.instantiate_weights()
   #     self.initializer=tf.random_normal_initializer(stddev=0.1)
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        if not os.path.exists(self.conf.logdir): 
            os.makedirs(self.conf.logdir)
        if not os.path.exists(self.conf.sampledir):
            os.makedirs(self.conf.sampledir)
        if self.conf.isTraining == True:
            self.configure_networks()
        else:
            self.configure_networks_test()
    
    def configure_networks(self):
        self.build_network()
    #    variables = tf.trainable_variables()
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, tf.train.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', update_ops=[])            
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary()


    def configure_networks_test(self):
        
        self.build_network_test()
       

        dic_layer = {1:self.out_layer1_2, 2: self.out_layer2_2, 3:self.out_layer3_2}
        dic_channel = {1:512, 2: 256, 3:128}
        self.v_0 = tf.placeholder(tf.float32, [None, dic_channel[self.conf.layer_num]], name="activations")
        self.x = tf.placeholder(tf.int32, name="corresponding_idx")
        self.left = tf.placeholder(tf.int32, name="left_side")
        self.right = tf.placeholder(tf.int32, name= "right_side")
        print('v0 has a shpae of', self.v_0.get_shape())
        layer = dic_layer[self.conf.layer_num]
        if self.conf.type == 'layer':
            print("Our target is a layer ================")
            self.target = layer
        if self.conf.type == 'sp_location':
            print("Our target is a sp_location ================")
            self.target = layer[:,self.x,:]
        if self.conf.type == 'channel':
            print("Our target is a channel ================")
            self.target = layer[:,:,self.conf.channel_num]
        if self.conf.type =='neuron':
            print("Our target is a neuron ================")
            self.target = layer[:,self.conf.x,self.conf.channel_num]
        print(self.target.get_shape())
        self.loss = self.get_optimized_loss()
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, tf.train.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=[self.initialized_x],update_ops=[])
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        trainable_vars = [var for var in trainable_vars if "optimized_input" not in var.name]
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.train_summary = self.config_summary_test()

    def get_optimized_loss(self, l2_lambda=0.002, l3_lambda= 0.01):
        if self.conf.type =='neuron':
            self.loss1 = tf.reduce_mean(self.target)
        elif self.conf.type =='sp_location':
            self.loss1 = tf.reduce_mean(tf.multiply(self.target, self.v_0))
        #    loss = tf.reduce_mean
        matrix = self.initialized_x[:,self.left:self.right,:]
        self.loss2 = tf.nn.l2_loss(matrix)*l2_lambda
        matrix_new = tf.multiply(matrix, matrix)
        print(matrix_new.shape)
        matrix_new = tf.reduce_sum(matrix_new, axis= 2)
        matrix_new =tf.sqrt(matrix_new)
        matrix_new = tf.expand_dims(matrix_new, axis= -1)
        matrix_final = tf.divide(matrix, matrix_new)
        similarity = tf.matmul(matrix_final, matrix_final, transpose_b=True)
        print("The similartiy shape is ", similarity.shape)
        self.loss3 = tf.reduce_mean(similarity)/2 *l3_lambda

        loss = -self.loss1+ self.loss2 - self.loss3
        return loss

    


    def build_network(self):

        self.input_x_embeded = tf.nn.embedding_lookup(self.Embedding, self.input_x) 
        print('input shape ',self.input_x_embeded.shape) 

  #      self.input_x_embeded0 = self.input_x_embeded
        self.mask0= tf.cast(tf.expand_dims(self.input_mask, -1), tf.float32)
    #    print('mask shape is ', self.mask0.shape)
        self.input_x_embeded = tf.multiply(self.input_x_embeded, self.mask0)

    #    print('after mask, the shape ',self.input_x_embeded.shape)

        with tf.variable_scope('Infer') as scope:
            self.logits= self.inference(self.input_x_embeded)

        print('the shape of logits ============', self.logits.get_shape())
        self.label = tf.argmax(self.input_y, axis=1) 
        print('the size of label is ============', self.label)        
        self.loss = self.get_loss()
        self.prob = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=1)
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32),tf.cast(self.label, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def inference(self, input_x):
        params1 = {"inputs": input_x, "filters": self.conf.d_model, "kernel_size": 5, "padding": "same",
                "activation": None, "use_bias": True , "name": "conv_layer_1"}  
        self.out_layer1 = tf.layers.conv1d(**params1)
        self.out_layer1 = tf.layers.batch_normalization(self.out_layer1, axis=-1, momentum=0.99, center=True, 
                epsilon=1e-5, training= self.conf.isTraining)
        self.out_layer1 = tf.nn.relu(self.out_layer1)
        print('after the first conv layer, the shape1 ',self.out_layer1.shape) 
        params2 = {"inputs": self.out_layer1, "filters": self.conf.d_model/2, "kernel_size": 4, "padding": "same",
                "activation": None, "use_bias": True , "name": "conv_layer_2"}  
        self.out_layer2 = tf.layers.conv1d(**params2)
        self.out_layer2 = tf.layers.batch_normalization(self.out_layer2, axis=-1, momentum=0.99, center=True, 
                epsilon=1e-5, training= self.conf.isTraining)
        self.out_layer2 = tf.nn.relu(self.out_layer2)                
        print('after the second conv layer, the shap1e ',self.out_layer2.shape)         
        params3 = {"inputs": self.out_layer2, "filters": self.conf.d_model/4, "kernel_size": 3, "padding": "same",
                "activation": None, "use_bias": True , "name": "conv_layer_3"}         
        self.out_layer3 = tf.layers.conv1d(**params3)
        self.out_layer3 = tf.layers.batch_normalization(self.out_layer3, axis=-1, momentum=0.99, center=True, 
                epsilon=1e-5, training= self.conf.isTraining)
        self.out_layer3 = tf.nn.relu(self.out_layer3) 
        print('after the third conv layer, the shape1 ',self.out_layer3.shape)  #[batch, length, 128]
        out_layer3 = tf.expand_dims(self.out_layer3, 1)  #[batch, 1, length, 128]
        self.pooled_layer = tf.nn.max_pool(out_layer3, ksize=[1,1,self.conf.sequence_length,1], strides=[1,1,1,1], padding= 'VALID', name="max_pool") # [batch,1,1,d-dimen]
        print('After pooling, the shape is1 =========================', self.pooled_layer.get_shape())
        self.pooled_layer = tf.squeeze(self.pooled_layer, [1,2]) #[batch, 128]
        print('After squeezing, the shape is1 =================', self.pooled_layer.get_shape())
        self.logits = tf.contrib.layers.fully_connected(self.pooled_layer, self.conf.num_classes, scope='fully_connected',
            activation_fn=None)    
        return self.logits    


    def inference2(self, input_x2):
        params1 = {"inputs": input_x2, "filters": self.conf.d_model, "kernel_size": 5, "padding": "same",
                "activation": None, "use_bias": True , "name": "conv_layer_1"}  
        self.out_layer1_2 = tf.layers.conv1d(**params1)
        self.out_layer1_2 = tf.layers.batch_normalization(self.out_layer1_2, axis=-1, momentum=0.99, center=True, 
                epsilon=1e-5, training= self.conf.isTraining)
        self.out_layer1_2 = tf.nn.relu(self.out_layer1_2)
        print('after the first conv layer, the shape ',self.out_layer1_2.shape) 
        params2 = {"inputs": self.out_layer1_2, "filters": self.conf.d_model/2, "kernel_size": 4, "padding": "same",
                "activation": None, "use_bias": True , "name": "conv_layer_2"}  
        self.out_layer2_2 = tf.layers.conv1d(**params2)
        self.out_layer2_2 = tf.layers.batch_normalization(self.out_layer2_2, axis=-1, momentum=0.99, center=True, 
                epsilon=1e-5, training= self.conf.isTraining)
        self.out_layer2_2 = tf.nn.relu(self.out_layer2_2)                
        print('after the second conv layer, the shape ',self.out_layer2_2.shape)         
        params3 = {"inputs": self.out_layer2_2, "filters": self.conf.d_model/4, "kernel_size": 3, "padding": "same",
                "activation": None, "use_bias": True , "name": "conv_layer_3"}         
        self.out_layer3_2 = tf.layers.conv1d(**params3)
        self.out_layer3_2 = tf.layers.batch_normalization(self.out_layer3_2, axis=-1, momentum=0.99, center=True, 
                epsilon=1e-5, training= self.conf.isTraining)
        self.out_layer3_2 = tf.nn.relu(self.out_layer3_2) 
        print('after the third conv layer, the shape ',self.out_layer3_2.shape)  #[batch, length, 128]
        out_layer3_2 = tf.expand_dims(self.out_layer3_2, 1)  #[batch, 1, length, 128]
        self.pooled_layer_2 = tf.nn.max_pool(out_layer3_2, ksize=[1,1,self.conf.sequence_length,1], strides=[1,1,1,1], padding= 'VALID', name="max_pool") # [batch,1,1,d-dimen]
        print('After pooling, the shape is =========================', self.pooled_layer.get_shape())
        self.pooled_layer_2 = tf.squeeze(self.pooled_layer_2, [1,2]) #[batch, 128]
        print('After squeezing, the shape is =================', self.pooled_layer_2.get_shape())
        self.logits_2 = tf.contrib.layers.fully_connected(self.pooled_layer_2, self.conf.num_classes, scope='fully_connected',
            activation_fn=None)    
        return self.logits_2           
        
        
    def build_network_test(self):
        
        self.input_x_embeded = tf.nn.embedding_lookup(self.Embedding, self.input_x) 
     #   self.input_x_embeded2 = tf.nn.embedding_lookup(self.Embedding, self.input_x2) 
        self.mask0= tf.cast(tf.expand_dims(self.input_mask, -1), tf.float32)
    #    print('mask shape is ', self.mask0.shape)
        self.input_x_embeded = tf.multiply(self.input_x_embeded, self.mask0)


        with tf.variable_scope('Infer') as scope:
            self.logits1= self.inference(self.input_x_embeded)

        with tf.variable_scope('Infer', reuse=True) as scope:
            self.logits2 = self.inference2(self.initialized_x)
         #   self.logits2 = self.inference2(self.input_x_embeded2)

        print('the shape of logits ============', self.logits.get_shape())
  #      self.prediction = tf.argmax(self.logits1, axis=1)    
   #     self.prob2 = tf.nn.softmax(self.logits2)
        self.prediction1= tf.argmax(self.logits1, axis=1)                



    def config_summary_test(self):
        summarys = []                      
        summarys.append(tf.summary.scalar('/loss', self.loss))
    #    summarys.append(tf.summary.scalar('/accuracy', self.accuracy))
        summary = tf.summary.merge(summarys)
        return summary




    def config_summary(self):
        summarys = []                      
        summarys.append(tf.summary.scalar('/loss', self.loss))
        summarys.append(tf.summary.scalar('/accuracy', self.accuracy))
        summary = tf.summary.merge(summarys)
        return summary

    def get_loss(self, l2_lambda=0.00001):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= self.label, logits= self.logits)
        loss = tf.reduce_mean(loss)
        loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() 
            if ('bias' not in v.name ) and ('alpha' not in v.name)]) * l2_lambda
        return loss+loss_l2


    def instantiate_weights(self):
        """define all weights here"""
        with tf.variable_scope("embedding_projection"):  # embedding matrix
          #  self.Embedding = tf.get_variable("Embedding", shape=[self.conf.vocab_size, self.conf.embed_size], initializer=self.initializer, trainable=False)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.conf.vocab_size, self.conf.embed_size], initializer=self.initializer)
    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
    
    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        iterations = 1
        if self.conf.checkpoint >0:
            print('=======Now load the model===============')
            self.reload(self.conf.checkpoint)

        
        data = data_reader()

        # use word2vec pretrained embedding
        vocabulary = data.vocab_processor.vocabulary_
        initW = data.load_word2vec_embedding(vocabulary, self.conf.path_embedding, True)
     #   self.sess.run(self.Embedding.assign(initW))
        print("finish load the pre_trained embedding==============")
        
        max_epoch = int (self.conf.max_epoch - (self.conf.checkpoint)/ (len(data.y_train)/self.conf.batch_size))
        print("Each epoch we have the number of batches", int (len(data.y_train)/self.conf.batch_size))
        max_acc= 0
        number_epoch = 0
        for epoch in range(number_epoch):
            pbar = ProgressBar()            
            for i in pbar(range(int (len(data.y_train)/self.conf.batch_size))):            
                x, y, mask= data.next_batch(self.conf.batch_size)
            #   x, y, mask= data.next_batch(1)
          #      feed_dict= {self.input_x:x, self.input_y: y, self.dropout_keep_prob:0.5}
                feed_dict= {self.input_x:x, self.input_y: y, self.input_mask: mask}
                _, loss, summary, accuracy , logits,label = self.sess.run([self.train_op, self.loss, self.train_summary, self.accuracy, self.logits, self.label], feed_dict= feed_dict)

           #     time.sleep(20)
            #    print("training loss is =============", loss, "  the acc is =============", accuracy)
                if iterations %self.conf.summary_step == 1:
                    self.save_summary(summary, iterations+self.conf.checkpoint)

                if iterations %self.conf.save_step == 0:
                    self.save(iterations+self.conf.checkpoint)
                iterations = iterations + 1
     #   if epoch % self.conf.eva_step == 1:
        overall_acc=0
        data.reset()
        for test_i in range(1):
            x_test, y_test, mask_test = data.read_visual_data()
        #    feed_dict2= {self.input_x:x_test, self.input_y: y_test, self.dropout_keep_prob:1}
            feed_dict2= {self.input_x:x_test, self.input_y: y_test, self.input_mask: mask_test}
            acc, test_loss, prediction = self.sess.run([self.accuracy, self.loss, self.prediction], feed_dict= feed_dict2)
            print(prediction.shape)
            np.save('visual_prediction.npy', prediction)
            overall_acc = overall_acc +acc
        #    print(test_i,' and',acc,'and ', overall_acc)
        overall_acc = overall_acc/1
        print(overall_acc)
            # if overall_acc > max_acc:
            #     max_acc = overall_acc
            #     number_epoch = epoch
            #     self.save(66666)
            # print("For the epoch  ", epoch, " test acc is  ", overall_acc, "==========the max_acc is", max_acc, "and the epoch is =====", number_epoch)




    def test(self):
        if self.conf.checkpoint >0:
            print('=======Now load the model===============')
            self.reload(self.conf.checkpoint) 
        data = data_reader()

        path = './result.txt'
        
        path2 = './new_representation.txt'
        
        Embedding = self.sess.run(self.Embedding)
        for i in range(1):
            print(i,"=================>>>>>>>>>>>>>>>>>>>>>>")
            f = open(path, "a")
            f2 =open(path2, "a")
            x, x_text, test_mask,y  = data.get_test_x()
            print(x_text)
            f.write("This is test example " +str(i)+ '\n')
    #     x2 = data.get_test_x2()
        #    print(x-x2)
            feed_dict = {self.input_x: x, self.input_mask:test_mask}
        #    r1, r2, x1, x2 = self.sess.run([self.logits1, self.logits2, self.initialized_x , self.input_x_embeded], feed_dict=feed_dict)
        #   print(r1)
        #  print(r2)
        #   time.sleep(100)
            
            # np.save('embedding.npy', Embedding)
            # print("Embedding is saved")
            # time.sleep(10)        


            dic_layer = {1:self.out_layer1, 2: self.out_layer2, 3:self.out_layer3}
            layer, clss = self.sess.run([dic_layer[self.conf.layer_num], self.prediction1],feed_dict=feed_dict)
        #    print(clss)
    #       print(self.logits2[:,clss:clss+1].shape)
    #       time.sleep(20)
            gradients = self.sess.run(tf.gradients(self.logits1[:,clss], dic_layer[self.conf.layer_num]), feed_dict=feed_dict)[0]
            # gradients2 = self.sess.run(tf.gradients(self.logits1[:,clss], dic_layer[self.conf.layer_num]), feed_dict=feed_dict)
            # print(gradients[0][:,0:,:])
            # print(gradients2[0][:,0:,:])
            #time.sleep(10)

            attribution_dict = {}
            for i in range(self.conf.sequence_length):
                if self.conf.type == 'sp_location':
                #    print("Our target is a sp_location ================")
                    activations_target  = layer[:,i,:]  
                    gradient_sp = gradients[:,i,:]
                    attribution_dict[i]=np.inner(activations_target, gradient_sp)[0][0]
                #    print(attribution_dict[i])
            attribution_sorted = sorted(attribution_dict.items(), key=operator.itemgetter(1), reverse=True)
         #   print(attribution_sorted)
            dict_res = {0: 'World', 1: 'Sports', 2:'Business', 3:'Sci/Tech'}
            attribution_k = 3
            f2.write(str(np.argmax(y, axis=1)[0])+', ')
            res_name = []
            res_vec = []
            for i in range(attribution_k):
            #    print("This is the top ", i+1,  "contribution position!~!!!!")
                item = attribution_sorted[i]
                idx= item[0]
                # to do : get the target and optimize
                activations_ = layer[:,idx,:]                
            #    print(type(activations_target))
           #     print(activations_target.shape)   

        #    print(Embedding[0]-a)
        #   print(Embedding[1]-b)
    #     print(Embedding[2]-c)
        #    time.sleep(100)
        #      self.conf.x= idx
                x_0 = 0
         #       print(type(idx))
        #     time.sleep(20)
                feed_dict2 = {self.v_0: activations_, self.x: idx}
                m = np.squeeze(self.sess.run(tf.gradients(self.target, self.initialized_x), feed_dict=feed_dict2)[0], axis=0)
                m = np.mean(m, axis = 1)
        #    print(m[0], m[1], m[194])
        #   print(m[195])
                start, end = self.get_index(m)
          #      print(start, "=================", end)
        #     input_image = np.squeeze(input_image, axis= 0)            
                feed_dict3 = {self.v_0: activations_, self.x: idx, self.left:start, self.right:end+1}
            #    print("The shape of v0 is", activations_)
        #      print(gradients[:,idx,:])
        #      print(np.inner(activations_ , gradients[:,idx,:]))
          #      time.sleep(20)
                for i in range(self.conf.opt_iter):
                    
                    _ ,input_image, loss1, loss2, loss3= self.sess.run([self.train_op, self.initialized_x, self.loss1, self.loss2, self.loss3], feed_dict=feed_dict3) 
            #        print(type(input_image-x_0))
                    diff = np.mean(np.absolute(np.squeeze(input_image-x_0, axis=0)))
                    x_0 = input_image
           #         print("In the iteration ", i, "the loss1 is ", loss1, "the loss2 is ", loss2,"loss3 is  ",loss3, "and the diff is ", diff)
                
                m = np.squeeze(self.sess.run(tf.gradients(self.target, self.initialized_x), feed_dict=feed_dict2)[0], axis=0)
                m = np.mean(m, axis = 1)
        #    print(m[0], m[1], m[194])
        #   print(m[195])
                start, end = self.get_index(m)
          #      print(start, "=================", end)
                input_image = np.squeeze(input_image, axis= 0)
    #      start, end = 
    #     helper_ = helper()
                result = [] 
                k = 10

                # print(x_text)
                # for i in range(start, end+1):
                #     test = input_image[i,:]
                #     lst =data.find_neighbors(Embedding, test, k)
                #     print("This is optimiazed results for word: ", i)
                #     for item in lst:
                        
                #         print(item[0], item[1])
                #     result.append((test, lst))   
            #     input_image_abs = np.absolute(input_image)    
            #     print(input_image_abs.shape) 
            #     row_sums = input_image_abs.sum(axis = 1)
            #     print(row_sums.shape)
            #     test = input_image/row_sums[:,np.newaxis]
            #     print(test.shape)
            #     print(test[start:end+1,:].shape)
                
            #     test_final = np.amax(test[start:end+1,:], axis=0)

            # #    test = np.amax(input_image[start:end+1,:], axis=0)
            #     print(test_final.shape)
            #     time.sleep(20)
                test = 0
                for i in range(start, end+1):
                    test= test + input_image[i,:]
                test = test/(end+1-start)

            
            #helper_ =helper()
            #lst = helper_.find_neighbors(test, k)

                lst =data.find_neighbors(Embedding, test, k)
                x_new = [[i for i in x[0] if i!=0]]
                f.write(str(data.convert_to_text(x_new))+"\n")
                f.write('\n')
           #     print(x_text)
          #      print(data.convert_to_text(x[:,start:end+1]))
                f.write(str(data.convert_to_text(x[:,start:end+1]))+'\n')
                f.write('\n')
           #     print(clss.item(0))
          #      print(np.argmax(y, axis=1))
           #     print("Our prediction is", dict_res[clss.item(0)], "and the groundtruth is ", dict_res[np.argmax(y, axis=1)[0]])
                f.write("Our prediction is "+str(dict_res[clss.item(0)])+ " the groundtruth is "+ str(dict_res[np.argmax(y, axis=1)[0]])+'\n')
                f.write('\n')
          #      print(idx)
    #      print(len(lst))
    #      time.sleep(10)
                for item in lst:
            #        print(item[0], item[1])
                    f.write(str(item[0])+' '+str(item[1])+'\n')
                result.append((test, lst))
                for i in range(5):
                    item = lst[i]
                    f2.write(str(item[0])+' ')
                
                f.write('\n')
                f.write('\n')
             #   result.append((test, lst))
                for i in range(len(result)):
                    target_name = 'targetword'+str(i)
                    vec = result[i][0]
                    res_name.append(target_name)
                    res_vec.append(vec)
                    for c in result[i][1]:
                    #  res_vec.append(helper_.get_vector(c[1]))
                        res_name.append(c[0].replace('_', ' '))
                        res_vec.append(c[2])
            print(res_name)
            f2.write('\n')
            f.close()
            f2.close()
            sio.savemat("vector.mat",{"vec":res_vec, "vec_name": res_name})
        # res_name = []
        # res_vec = []
        # for i in range(len(result)):
        #     target_name = 'targetword'+str(i)
        #     vec = result[i][0]
        #     res_name.append(target_name)
        #     res_vec.append(vec)
        #     for c in result[i][1]:
        #         res_vec.append(helper_.get_vector(c[1]))
        #         res_name.append(c[0].replace('_', ' '))
        #       #  res_vec.append(c[2])
                
        
        # print(res_name)
        
        # sio.savemat("vector.mat",{"vec":res_vec, "vec_name": res_name})

        



        
        
    #    data = data_reader()
        
    def get_index(self, m):
        left = 0
        right = len(m)-1
        print(right)
        while(m[left]==0):
            left= left+1
        while(m[right]==0):
            right =right -1
        return[left,right]



        


        

    def reload(self, epoch):
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
     #   print(model_path)
        model_path = checkpoint_path +'-'+str(epoch)
        print(model_path)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return       
        self.saver.restore(self.sess, model_path)
        print("model load successfully===================")

