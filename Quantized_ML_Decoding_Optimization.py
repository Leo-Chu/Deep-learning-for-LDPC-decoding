# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:28:08 2019

@author: user
"""

import numpy as np
import tensorflow as tf
import datetime
import DataIO
import os
import Quantized_ML_Decoder

class BP_Training:
    def __init__(self, train_config_in, top_config_in, code_in):
        #config
        self.train_config = train_config_in
        self.top_config = top_config_in
        self.code = code_in
        #para_alpha
        self.alpha_name = {}
        self.alpha = {}
        self.best_alpha = {}
        self.assign_best_alpha = {}
        #para_beta
        self.beta_name = {}
        self.beta = {}
        self.best_beta = {}
        self.assign_best_beta = {}
    
    def build_network(self, bp_decoder, built_for_training=False):   
        #x_in
        x_in = tf.placeholder(tf.float32, [None, self.train_config.training_minibatch_size])
        xe_0 = tf.placeholder(tf.float32, [self.train_config.feature_length, self.train_config.training_minibatch_size])
        #para_name
        self.alpha_name = format("alpha")
        self.beta_name = format("beta")
                
        #alpha&beta
        if built_for_training:
            self.alpha= tf.get_variable(name=self.alpha_name, shape=[1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.best_alpha = tf.Variable(tf.ones(1, tf.float32), dtype=tf.float32)
            self.assign_best_alpha = self.best_alpha.assign(self.alpha)
            self.beta = tf.get_variable(name=self.beta_name, shape=[1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.best_beta = tf.Variable(tf.ones(1, tf.float32), dtype=tf.float32)
            self.assign_best_beta = self.best_beta.assign(self.beta)
        else:
#            self.alpha = tf.round(tf.Variable(5*tf.ones([1]), dtype=tf.float32, name=self.alpha_name)*10)/10
            self.alpha = tf.Variable(1*tf.ones([1]), dtype=tf.float32, name=self.alpha_name)
            self.best_alpha = tf.Variable(tf.ones(1, tf.float32), dtype=tf.float32)
            self.assign_best_alpha = self.best_alpha.assign(self.alpha)
            self.beta = tf.Variable(1.0*tf.ones([1]), dtype=tf.float32, name=self.beta_name)
#            self.beta = tf.Variable(0*tf.ones([1]), dtype=tf.float32, name=self.beta_name, trainable=False)
            self.best_beta = tf.Variable(tf.ones(1, tf.float32), dtype=tf.float32)
            self.assign_best_beta = self.best_beta.assign(self.beta)
            #NN
            y_out = bp_decoder.one_nn_iteration(x_in, xe_0, self.alpha, self.beta)
        return x_in, xe_0, y_out
    
    def softsign(self, x_in):
        y_out = x_in/(tf.abs(x_in) + 0.01)
        return y_out
        
    def save_network_temporarily(self, sess_in):
        sess_in.run(self.assign_best_alpha)
        sess_in.run(self.assign_best_beta)
    
    def test_network_online(self, dataio, decoder_test, iteration, x_in, xe_0, y_label, orig_loss, loss_after_training, calc_org_loss, sess_in):
        # this function is used to test the network loss online when training network
        remain_samples = self.train_config.test_sample_num
        load_batch_size = self.train_config.test_minibatch_size
        ave_orig_loss = 0.0        
        ave_loss_after_train = 0.0
        while remain_samples > 0:
            if remain_samples < self.train_config.test_minibatch_size:
                load_batch_size = remain_samples
            batch_xs, batch_ys  = dataio.load_batch_for_test(load_batch_size)  # features, labels
            bp_out_xs, bp_out_xe0 = decoder_test.quantized_decode_before_nn(batch_xs, iteration, self.train_config.alpha, self.train_config.beta)
            if calc_org_loss:
                orig_loss_value, loss_after_training_value = sess_in.run([orig_loss, loss_after_training], feed_dict={x_in: bp_out_xs, xe_0: bp_out_xe0, y_label: batch_ys})
                ave_orig_loss += orig_loss_value * load_batch_size
            else:
                loss_after_training_value = sess_in.run(loss_after_training, feed_dict={x_in: bp_out_xs, xe_0: bp_out_xe0, y_label: batch_ys})
            remain_samples -= load_batch_size
            ave_loss_after_train += loss_after_training_value * load_batch_size
            
        if calc_org_loss:
            ave_orig_loss /= np.double(self.train_config.test_sample_num)
        ave_loss_after_train /= np.double(self.train_config.test_sample_num)
        if calc_org_loss:
            print("Orig loss: %f, Test loss: %f" % (ave_orig_loss, ave_loss_after_train))
        return ave_orig_loss, ave_loss_after_train
    
    def cal_training_loss(self, y_out, y_label):   
        y_out = tf.to_double(y_out)
        y_label = tf.to_double(y_label)
        y_out1 = tf.div(1-self.softsign(tf.matmul(self.top_config.D,y_out)), 2)
        training_loss = tf.reduce_mean(tf.square(y_out1-tf.transpose(y_label)))
        return training_loss
    
    def train_network(self, model_id, test_iter, test_SNR):
        start = datetime.datetime.now()
        bp_decoder = Quantized_ML_Decoder.BP_NetDecoder(self.code.H_matrix, self.train_config.training_minibatch_size)
        x_in, xe_0, y_out = self.build_network(bp_decoder, False)
        y_label = tf.placeholder(tf.float32, [self.train_config.training_minibatch_size, self.train_config.label_length])
        
        training_loss = self.cal_training_loss(y_out, y_label)
        test_loss = training_loss    
        orig_loss_for_test = self.cal_training_loss(xe_0, y_label)
        
        # SGD_Adam
        train_step = tf.train.AdamOptimizer().minimize(training_loss)
        
        # init operation
        init = tf.global_variables_initializer()

        # create a session
        sess = tf.Session()     
        for SNR in test_SNR:
            print("Training for SNR = %.1f" % SNR)
            training_feature_file = format('%s%s/feature_%d_%d_%d_%.1f.dat' % (self.train_config.training_data_folder, self.top_config.channel, self.train_config.feature_length, self.train_config.label_length, self.train_config.training_minibatch_size,SNR))
            training_label_file = format('%s%s/label_%d_%d_%d_%.1f.dat' % (self.train_config.training_data_folder, self.top_config.channel, self.train_config.feature_length, self.train_config.label_length, self.train_config.training_minibatch_size,SNR))
            test_feature_file = format('%s%s/feature_%d_%d_%d_%.1f.dat' % (self.train_config.test_data_folder, self.top_config.channel, self.train_config.feature_length, self.train_config.label_length, self.train_config.test_minibatch_size,SNR))
            test_label_file = format('%s%s/label_%d_%d_%d_%.1f.dat' % (self.train_config.test_data_folder, self.top_config.channel, self.train_config.feature_length, self.train_config.label_length, self.train_config.test_minibatch_size,SNR))
            dataio_train = DataIO.TrainingDataIO(training_feature_file, training_label_file, self.train_config.training_sample_num, self.train_config.feature_length, self.train_config.label_length)
            dataio_test = DataIO.TestDataIO(test_feature_file, test_label_file, self.train_config.test_sample_num, self.train_config.feature_length, self.train_config.label_length) 
            for iteration in range(0, test_iter):
                start1 = datetime.datetime.now()
                sess.run(init)
                # calculate the loss before training and assign it to min_loss
                ave_orig_loss, min_loss = self.test_network_online(dataio_test, bp_decoder, iteration, x_in, xe_0, y_label, orig_loss_for_test, test_loss, True, sess)
        
                self.save_network_temporarily(sess)
                # Train
                count = 0
                epoch = 0
                print('Iteration\tBest alpha\tBest beta\tCurrent loss\tCurrent alpha\tCurrent beta')
        
                alpha_set = []
                beta_set = []
                while epoch < self.train_config.epoch_num:
                    epoch += 1
                    batch_xs, batch_ys = dataio_train.load_next_mini_batch(self.train_config.training_minibatch_size)
                    llr_into_nn_net, xe0_into_nn_net = bp_decoder.quantized_decode_before_nn(batch_xs, iteration, self.train_config.alpha, self.train_config.beta)
                    sess.run([train_step], feed_dict={x_in: llr_into_nn_net, xe_0: xe0_into_nn_net, y_label: batch_ys})
                    a,b = sess.run([self.alpha, self.beta])
                    alpha_set.append(a)
                    beta_set.append(b)
                    if epoch % 100 == 0 or epoch == self.train_config.epoch_num:
                        _, ave_loss_after_train = self.test_network_online(dataio_test, bp_decoder, iteration, x_in, xe_0, y_label, orig_loss_for_test, test_loss, False, sess)
                        if ave_loss_after_train < min_loss:
                            print('%d\t\t%f\t%f\t%f\t%f\t%f' % (epoch, sess.run(self.best_alpha), sess.run(self.best_beta), ave_loss_after_train, sess.run(self.alpha), sess.run(self.beta)))
                            min_loss = ave_loss_after_train
                            self.save_network_temporarily(sess)
                            count = 0
                        else:
                            print('%d\t\t%f\t%f\t%f\t%f\t%f' % (epoch, sess.run(self.best_alpha), sess.run(self.best_beta), ave_loss_after_train, sess.run(self.alpha), sess.run(self.beta)))
                            count += 1
                            if count >= 8:  # no patience
                                break
                best_alpha = sess.run(self.best_alpha)
                best_beta = sess.run(self.best_beta)
                self.train_config.alpha[iteration] = best_alpha
                self.train_config.beta[iteration] = best_beta
                para_file = format('%sSNR%.1f_Iter%d.txt' % (self.top_config.results_folder, SNR, iteration+1))
                np.savetxt(para_file,np.vstack((self.train_config.alpha, self.train_config.beta)))
                end1 = datetime.datetime.now()
                print('Used time for %dth training: %ds'% (iteration+1, (end1-start1).seconds))
                print('\n')
        sess.close()
        end = datetime.datetime.now()
        print('Used time for training: %ds'% (end-start).seconds)