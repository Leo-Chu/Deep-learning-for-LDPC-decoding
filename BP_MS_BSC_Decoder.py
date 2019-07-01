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
        self.max_row_weight = np.max(self.H_sum_row)
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
        return H_x_to_xe0, H_sumV_to_C, H_xe_v_sumc_to_y

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
        H_sumC_to_V = np.matmul(H_sum_by_C_to_V, map_H_line_to_row)      
        H_sum_row = np.sum(H_sumC_to_V, axis=1)
        H_sum_row = H_sum_row.astype(int)
        max_row_weight = np.max(H_sum_row)
        H_minC_to_V = np.zeros([count, max_row_weight])
        ii, jj = np.nonzero(H_sumC_to_V)
        h_min_segmentIDs = np.zeros([len(jj)])
        head_index = np.cumsum(H_sum_row) - H_sum_row
        end_index = np.cumsum(H_sum_row)
        head_index = head_index.astype(int)
        end_index = end_index.astype(int)
        for k in range(0, count):
            H_minC_to_V[k, 0:H_sum_row[k]] = jj[head_index[k]:end_index[k]]
            h_min_segmentIDs[head_index[k]:end_index[k]] = k
        h_minC_to_V = jj
        print("return Matrics C-V successfully!\n")
        return H_sumC_to_V, H_minC_to_V.astype(int), H_sum_row, h_minC_to_V.astype(int), h_min_segmentIDs.astype(int)


class BP_NetDecoder:
    def __init__(self, H, batch_size, alpha, beta):
        self.check_matrix = H
        _, self.v_node_num = np.shape(H)
        self.alpha = alpha
        self.beta = beta
        ii, jj = np.nonzero(H)
        loc_nzero_row = np.array([ii, jj])
        self.num_all_edges = np.size(loc_nzero_row[1, :])
        #get MatrixCV and MatrixVC
        gm1 = GetMatrixForBPNet(H[:, :], loc_nzero_row)
        self.H_sumC_to_V, self.H_minC_to_V, self.H_sum_row, self.h_minC_to_V, self.h_min_segmentIDs = gm1.get_Matrix_CV()
        self.H_x_to_xe0, self.H_sumV_to_C, self.H_xe_v_sumc_to_y = gm1.get_Matrix_VC()
        #LLR
        self.batch_size = batch_size
        self.llr_placeholder0 = tf.placeholder(tf.float32, [batch_size, self.v_node_num])
        
        self.llr_into_bp_net, self.xe_v2c_pre_iter_assign, self.start_next_iteration, self.dec_out = self.build_network()
        
        self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_placeholder0))  # transpose the llr matrix to adapt to the matrix operation in BP net decoder.
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session() # open a session
        print('Open a tf session!')
        self.sess.run(init)

    def __del__(self):
        self.sess.close()
        print('Close a tf session!') 
    
    def one_bp_iteration(self, xe_v2c_pre_iter, H_sumC_to_V, H_sumV_to_C, h_minC_to_V, h_min_segmentIDs, xe_0):
        xe_sign = tf.to_float(tf.sign(xe_v2c_pre_iter))
        xe_sum_log_img = tf.matmul(H_sumC_to_V, tf.multiply(tf.truediv((1 - xe_sign), [2.0]), [3.1415926]))
        xe_sum_log_complex = tf.complex(tf.zeros([self.num_all_edges, self.batch_size]), xe_sum_log_img)
        xe_product = tf.real(tf.exp(xe_sum_log_complex))
        xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        xe_pd_modified = tf.add(xe_product, xe_product_temp)
        xe_v_minc = tf.zeros([self.num_all_edges, self.batch_size], dtype = tf.float32)
        xe_v_minc = tf.segment_min(tf.abs(tf.gather(xe_v2c_pre_iter, h_minC_to_V)), h_min_segmentIDs)
        xe_v_sumc = tf.multiply(xe_pd_modified, tf.maximum((self.alpha*xe_v_minc+self.beta), tf.zeros([self.num_all_edges, self.batch_size], dtype=tf.float32)))
        xe_c_sumv = tf.add(xe_0, tf.matmul(H_sumV_to_C, xe_v_sumc))
        return xe_v_sumc, xe_c_sumv
    
    def build_network(self): # build the network for one BP iteration
        # BP initialization
        llr_into_bp_net = tf.Variable(np.ones([self.v_node_num, self.batch_size], dtype=np.float32))
        y_receive = tf.floordiv(1-llr_into_bp_net, 2)
        xe_0 = tf.matmul(self.H_x_to_xe0, llr_into_bp_net)
        
        xe_v2c_pre_iter = tf.Variable(np.ones([self.num_all_edges, self.batch_size], dtype=np.float32)) # the v->c messages of the previous iteration
        xe_v2c_pre_iter_assign = xe_v2c_pre_iter.assign(xe_0)

        # one iteration initialize
        H_sumC_to_V = tf.constant(self.H_sumC_to_V, dtype=tf.float32)
        H_sumV_to_C = tf.constant(self.H_sumV_to_C, dtype=tf.float32)
        h_minC_to_V = tf.constant(self.h_minC_to_V, dtype=tf.int32)
        h_min_segmentIDs = tf.constant(self.h_min_segmentIDs, dtype=tf.int32)
        
        # one iteration       
        xe_v_sumc, xe_c_sumv = self.one_bp_iteration(xe_v2c_pre_iter, H_sumC_to_V, H_sumV_to_C, h_minC_to_V, h_min_segmentIDs, xe_0)
        
        # start the next iteration
        start_next_iteration = xe_v2c_pre_iter.assign(xe_c_sumv)

        # get the final marginal probability and decoded results
        bp_out_llr = tf.add(llr_into_bp_net, tf.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc))
        bp_out_llr_nonzero = tf.equal(tf.abs(tf.sign(bp_out_llr)), 1)
        dec_out = tf.transpose(tf.where(bp_out_llr_nonzero, tf.floordiv(1-tf.sign(bp_out_llr), 2), y_receive))


        return llr_into_bp_net, xe_v2c_pre_iter_assign, start_next_iteration, dec_out

    def decode(self, llr_in, bp_iter_num):
        real_batch_size, num_v_node = np.shape(llr_in)
        if real_batch_size != self.batch_size:  # padding zeros
            llr_in = np.append(llr_in, np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32), 0)  # re-create an array and will not influence the value in
            # original llr array.
        self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder0: llr_in})
        self.sess.run(self.xe_v2c_pre_iter_assign)
        for iter in range(0, bp_iter_num-1):
            self.sess.run(self.start_next_iteration)
        y_dec = self.sess.run(self.dec_out) 
        if real_batch_size != self.batch_size:
            y_dec = y_dec[0:real_batch_size, :]
        return y_dec
