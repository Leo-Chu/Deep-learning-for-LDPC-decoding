import tensorflow as tf
import numpy as np


class GetMatrixForBPNet:
    # this class is to calculate the matrices used to perform BP process with matrix operation
    def __init__(self, test_H, loc_nzero_row):
        print("Construct the Matrics H class!\n")
        self.H = test_H
        self.m, self.n = np.shape(test_H)
        self.H_sum_line = np.sum(self.H, axis=0)
        self.H_sum_row = np.sum(self.H, axis=1)
        self.loc_nzero_row = loc_nzero_row
        self.num_all_edges = np.size(self.loc_nzero_row[1, :])

        self.loc_nzero1 = self.loc_nzero_row[1, :] * self.n + self.loc_nzero_row[0, :]
        self.loc_nzero2 = np.sort(self.loc_nzero1)
        self.loc_nzero_line = np.append([np.mod(self.loc_nzero2, self.n)], [self.loc_nzero2 // self.n], axis=0)
        self.loc_nzero4 = self.loc_nzero_line[0, :] * self.n + self.loc_nzero_line[1, :]
        self.loc_nzero5 = np.sort(self.loc_nzero4)

    ##########################################################################################################
    def get_Matrix_VC(self):
        H_x_to_xe0 = np.zeros([self.num_all_edges, self.n], np.float32)
        H_sum_by_V_to_C = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)
        H_xe_last_to_y = np.zeros([self.n, self.num_all_edges], dtype=np.float32)
        Map_row_to_line = np.zeros([self.num_all_edges, 1])

        for i in range(0, self.num_all_edges):
            Map_row_to_line[i] = np.where(self.loc_nzero1 == self.loc_nzero2[i])

        map_H_row_to_line = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        for i in range(0, self.num_all_edges):
            map_H_row_to_line[i, int(Map_row_to_line[i])] = 1

        count = 0
        for i in range(0, self.n):
            temp = count + self.H_sum_line[i]
            H_sum_by_V_to_C[count:temp, count:temp] = 1
            H_xe_last_to_y[i, count:temp] = 1
            H_x_to_xe0[count:temp, i] = 1
            for j in range(0, self.H_sum_line[i]):
                H_sum_by_V_to_C[count + j, count + j] = 0
            count = count + self.H_sum_line[i]
        print("return Matrics V-C successfully!\n")
        H_sumV_to_C = np.matmul(H_sum_by_V_to_C, map_H_row_to_line)
        H_xe_v_sumc_to_y = np.matmul(H_xe_last_to_y, map_H_row_to_line)
        return H_x_to_xe0, H_sumV_to_C, H_xe_v_sumc_to_y, H_xe_last_to_y

    ###################################################################################################
    def get_Matrix_CV(self):

        H_sum_by_C_to_V = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        Map_line_to_row = np.zeros([self.num_all_edges, 1])

        for i in range(0, self.num_all_edges):
            Map_line_to_row[i] = np.where(self.loc_nzero4 == self.loc_nzero5[i])

        map_H_line_to_row = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        for i in range(0, np.size(self.loc_nzero1)):
            map_H_line_to_row[i, int(Map_line_to_row[i])] = 1

        count = 0
        for i in range(0, self.m):
            temp = count + self.H_sum_row[i]
            H_sum_by_C_to_V[count:temp, count:temp] = 1
            for j in range(0, self.H_sum_row[i]):
                H_sum_by_C_to_V[count + j, count + j] = 0
            count = count + self.H_sum_row[i]
        print("return Matrics C-V successfully!\n")
        H_sumC_to_V = np.matmul(H_sum_by_C_to_V, map_H_line_to_row)
        return H_sumC_to_V


class BP_NetDecoder:
    def __init__(self, H, batch_size):
        self.check_matrix = H
        self.H_sum_line = np.sum(H, axis=0)
        _, self.v_node_num = np.shape(H)
        ii, jj = np.nonzero(H)
        loc_nzero_row = np.array([ii, jj])
        self.num_all_edges = np.size(loc_nzero_row[1, :])
        gm1 = GetMatrixForBPNet(H[:, :], loc_nzero_row)
        self.H_sumC_to_V = gm1.get_Matrix_CV()
        self.H_x_to_xe0, self.H_sumV_to_C, self.H_xe_v_sumc_to_y, self.H_xe_last_to_y = gm1.get_Matrix_VC()
        self.batch_size = batch_size
        self.llr_placeholder = tf.placeholder(tf.float32, [batch_size, self.v_node_num])
        self.llr_into_bp_net, self.xe_0, self.xe_v2c_pre_iter_assign, self.start_next_iteration, self.dec_out = self.build_network()
        self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_placeholder))  # transpose the llr matrix to adapt to the matrix operation in BP net decoder.

        init = tf.global_variables_initializer()
        self.sess = tf.Session() # open a session
        print('Open a tf session!')
        self.sess.run(init)

    def __del__(self):
        self.sess.close()
        print('Close a tf session!')
        
    def get_final_marginalization(self, y_dec):
        _, batch_size = np.shape(y_dec)
        syndrome_weight = np.sum(np.mod(np.matmul(self.check_matrix, y_dec), 2), axis=0)
        final_marginalization =np.reshape(1-np.digitize(syndrome_weight, np.array([0]), right=True), (self.batch_size,1))
        return final_marginalization

    def atanh(self, x):
        x1 = tf.add(1.0, x)
        x2 = tf.subtract((1.0), x)
        x3 = tf.divide(x1, x2)
        x4 = tf.log(x3)
        return tf.divide(x4, (2.0))

    def one_bp_iteration(self, xe_v2c_pre_iter, H_sumC_to_V, H_sumV_to_C, xe_0):
        xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))
        xe_tanh = tf.to_float(xe_tanh)
        xe_tanh_temp = tf.sign(xe_tanh)
        xe_sum_log_img = tf.matmul(H_sumC_to_V, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]), [3.1415926]))
        xe_sum_log_real = tf.matmul(H_sumC_to_V, tf.log(1e-8 + tf.abs(xe_tanh)))
        xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
        xe_product = tf.real(tf.exp(xe_sum_log_complex))
        xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        xe_pd_modified = tf.add(xe_product, xe_product_temp)
        xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
        xe_c_sumv = tf.add(xe_0, tf.matmul(H_sumV_to_C, xe_v_sumc))
        return xe_v_sumc, xe_c_sumv

    def build_network(self): # build the network for one BP iteration
        # BP initialization
        llr_into_bp_net = tf.Variable(np.ones([self.v_node_num, self.batch_size], dtype=np.float32))
        xe_0 = tf.matmul(self.H_x_to_xe0, llr_into_bp_net)
        xe_v2c_pre_iter = tf.Variable(np.ones([self.num_all_edges, self.batch_size], dtype=np.float32)) # the v->c messages of the previous iteration
        xe_v2c_pre_iter_assign = xe_v2c_pre_iter.assign(xe_0)

        # one iteration
        H_sumC_to_V = tf.constant(self.H_sumC_to_V, dtype=tf.float32)
        H_sumV_to_C = tf.constant(self.H_sumV_to_C, dtype=tf.float32)
        xe_v_sumc, xe_c_sumv = self.one_bp_iteration(xe_v2c_pre_iter, H_sumC_to_V, H_sumV_to_C, xe_0)

        # start the next iteration
        start_next_iteration = xe_v2c_pre_iter.assign(xe_c_sumv)

        # get the final marginal probability and decoded results
        bp_out_llr = tf.add(llr_into_bp_net, tf.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc))
        dec_out = tf.transpose(tf.floordiv(1-tf.to_int32(tf.sign(bp_out_llr)), 2))
#        
#        bp_out_llr = tf.add(llr_into_bp_net, tf.transpose(tf.transpose(tf.matmul(self.H_xe_last_to_y, (xe_c_sumv-xe_0)))/(self.H_sum_line-1)))
#        dec_out = tf.transpose(tf.floordiv(1-tf.to_int32(tf.sign(bp_out_llr)), 2))
        
        return llr_into_bp_net, xe_0, xe_v2c_pre_iter_assign, start_next_iteration, dec_out

    def decode(self, llr_in, bp_iter_num):
        real_batch_size, num_v_node = np.shape(llr_in)
        if real_batch_size != self.batch_size:  # padding zeros
            llr_in = np.append(llr_in, np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32), 0)  # re-create an array and will not influence the value in
            # original llr array.
        self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})
        self.sess.run(self.xe_v2c_pre_iter_assign)
        for iter in range(0, bp_iter_num-1):
            self.sess.run(self.start_next_iteration)
        y_dec = self.sess.run(self.dec_out)
        if real_batch_size != self.batch_size:
            y_dec = y_dec[0:real_batch_size, :]
        return y_dec

#    def decode(self, llr_in, bp_iter_num):
#        real_batch_size, num_v_node = np.shape(llr_in)
#        y_dec = np.zeros([self.batch_size,self.v_node_num],dtype = np.float32)
#        temp0_final_marginalization = np.zeros([self.batch_size,1])
#        temp1_final_marginalization = np.zeros([self.batch_size,1])
#        if real_batch_size != self.batch_size:  # padding zeros
#            llr_in = np.append(llr_in, np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32), 0)  # re-create an array and will not influence the value in
#            # original llr array.
#        self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})
#        self.sess.run(self.xe_v2c_pre_iter_assign)
#        for iter in range(0, bp_iter_num-1):
#            temp_y_dec = self.sess.run(self.dec_out)           
#            final_marginalization = self.get_final_marginalization(np.transpose(temp_y_dec))     
#            temp1_final_marginalization = np.multiply((1-temp0_final_marginalization), final_marginalization)
#            temp0_final_marginalization = temp0_final_marginalization + temp1_final_marginalization
#            y_dec = y_dec + np.multiply(temp1_final_marginalization, temp_y_dec)
#            self.sess.run(self.start_next_iteration)           
#        temp_y_dec = self.sess.run(self.dec_out)
#        y_dec = y_dec + np.multiply(temp_y_dec, (1-temp0_final_marginalization))
#        if real_batch_size != self.batch_size:
#            y_dec = y_dec[0:real_batch_size, :]
#        return y_dec