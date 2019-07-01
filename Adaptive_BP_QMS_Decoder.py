import tensorflow as tf
import numpy as np


import tensorflow as tf
import numpy as np


class GetMatrixForBPNet:
    # this class is to calculate the matrices used to perform BP process with matrix operation
    def __init__(self, H, loc_nzero_row, ):
        print("Construct the Matrics H class!\n")
        #dim para
        self.H = H
        self.m, self.n = np.shape(H)
        #H
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
        H_sum_row = np.sum(H_sumC_to_V, axis=1).astype(int)
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
        return H_sumC_to_V, H_minC_to_V.astype(int), h_minC_to_V.astype(int), h_min_segmentIDs.astype(int), map_H_line_to_row.astype(int)
    ##########################################################################################################

class BP_NetDecoder:
    def __init__(self, H, batch_size, beta_set, syndrome_weight_determining):
        self.batch_size = batch_size
        self.check_matrix = H
        _, self.v_node_num = np.shape(H)
        ii, jj = np.nonzero(H)
        loc_nzero_row = np.array([ii, jj])
        self.num_all_edges = np.size(loc_nzero_row[1, :])
        
        #quantizer
        self.nambda = 1.5
        self.label = np.array([[1/3,1]])*self.nambda
        self.beta_set = beta_set
        self.syndrome_weight_determining = syndrome_weight_determining
        self.quantizer_num = np.size(self.beta_set)
        
        #get H&P MatrixCV and MatrixVC
        gm = GetMatrixForBPNet(self.check_matrix, loc_nzero_row) 
        self.H_sumC_to_V, self.H_minC_to_V, self.h_minC_to_V, self.h_min_segmentIDs, self.map_H_line_to_row = gm.get_Matrix_CV()
        self.H_x_to_xe0, self.H_sumV_to_C, self.H_xe_v_sumc_to_y = gm.get_Matrix_VC()
        
        #LLR
        self.llr_placeholder = tf.placeholder(tf.float32, [batch_size, self.v_node_num])
        
        self.llr_into_bp_net, self.xe_c2v_pre_iter_assign, self.start_next_iteration, self.dec_out, self.quantizer_index, self.syndrome_weight, self.xe_c_sumv_nonq, self.xe_c_sumv = self.build_network()
        self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_placeholder))  # transpose the llr matrix to adapt to the matrix operation in BP net decoder.
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session() # open a session
        print('Open a tf session!')
        self.sess.run(init)
        
    def __del__(self):
        self.sess.close()
        print('Close a tf session!')

    def quantizer(self, message, quantizer_index, sigma_square = 0.0004):
        quantizer_index_onehotvector = tf.transpose(tf.one_hot(quantizer_index, self.quantizer_num, dtype=tf.float64))
        beta = tf.matmul(self.beta_set, quantizer_index_onehotvector)
        label = tf.transpose(tf.to_double(tf.transpose(self.label)*beta))
        message = tf.to_double(tf.clip_by_value(message, -1, 1))
        a = label[:,0]*tf.exp(-tf.square(message-label[:,0])/2/sigma_square) + label[:,1]*tf.exp(-tf.square(message-label[:,1])/2/sigma_square) - label[:,0]*tf.exp(-tf.square(message+label[:,0])/2/sigma_square) - label[:,1]*tf.exp(-tf.square(message+label[:,1])/2/sigma_square)
        b = tf.exp(-tf.square(message-label[:,0])/2/sigma_square) + tf.exp(-tf.square(message-label[:,1])/2/sigma_square) + tf.exp(-tf.square(message+label[:,0])/2/sigma_square) + tf.exp(-tf.square(message+label[:,1])/2/sigma_square)
        quantized_message = tf.to_float(1/self.nambda*a/b)
        return quantized_message
    
    def get_quantizer_index(self, message, H_xe_v_sumc_to_y, llr_into_bp_net):
        y_dec = tf.floordiv(1-tf.sign(tf.add(llr_into_bp_net, tf.matmul(H_xe_v_sumc_to_y, message))), 2)
        syndrome_weight = tf.reduce_sum(tf.mod(tf.matmul(self.check_matrix, tf.to_int32(y_dec)), 2), axis=0)
        sy1 = tf.maximum(tf.sign(syndrome_weight-self.syndrome_weight_determining[0]),0)
        sy2 = tf.maximum(tf.sign(syndrome_weight-self.syndrome_weight_determining[1]),0)
        quantizer_index = sy1+sy2
        return tf.to_int32(quantizer_index), syndrome_weight

    def one_bp_iteration(self, xe_c2v_pre_iter, xe_0, llr_into_bp_net, H_sumC_to_V, h_minC_to_V, h_min_segmentIDs, H_sumV_to_C, H_xe_v_sumc_to_y):
        quantizer_index, syndrome_weight = self.get_quantizer_index(xe_c2v_pre_iter, H_xe_v_sumc_to_y, llr_into_bp_net)
        xe_c_sumv_nonq = tf.add(xe_0, tf.matmul(H_sumV_to_C, xe_c2v_pre_iter))
        xe_c_sumv = self.quantizer(tf.add(xe_0, tf.matmul(H_sumV_to_C, xe_c2v_pre_iter)), quantizer_index)
        xe_sign = tf.to_float(tf.sign(xe_c_sumv))
        xe_sum_log_img = tf.matmul(H_sumC_to_V, tf.multiply(tf.truediv((1 - xe_sign), [2.0]), [3.1415926]))
        xe_sum_log_complex = tf.complex(tf.zeros([self.num_all_edges, self.batch_size]), xe_sum_log_img)
        xe_product = tf.real(tf.exp(xe_sum_log_complex))
        xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        xe_pd_modified = tf.add(xe_product, xe_product_temp)
        xe_v_minc = tf.segment_min(tf.abs(tf.gather(xe_c_sumv, h_minC_to_V)), h_min_segmentIDs)
        xe_v_sumc = tf.multiply(xe_pd_modified, xe_v_minc)
        return xe_v_sumc, quantizer_index, syndrome_weight, xe_c_sumv_nonq, xe_c_sumv
    
    def build_network(self): # build the network for one BP iteration
        # BP initialization
        llr_into_bp_net = tf.Variable(np.ones([self.v_node_num, self.batch_size], dtype=np.float32))
        xe_0 = tf.matmul(self.H_x_to_xe0, llr_into_bp_net)
        
        xe_c2v_pre_iter = tf.Variable(np.ones([self.num_all_edges, self.batch_size], dtype=np.float32)) # the v->c messages of the previous iteration 
        xe_c2v_pre_iter_assign = xe_c2v_pre_iter.assign(tf.zeros([self.num_all_edges, self.batch_size],dtype=tf.float32))
        # one iteration initialize
        H_sumC_to_V = tf.constant(self.H_sumC_to_V, dtype=tf.float32)
        H_sumV_to_C = tf.constant(self.H_sumV_to_C, dtype=tf.float32)
        h_minC_to_V = tf.constant(self.h_minC_to_V, dtype=tf.int32)
        h_min_segmentIDs = tf.constant(self.h_min_segmentIDs, dtype=tf.int32)
        H_xe_v_sumc_to_y = tf.constant(self.H_xe_v_sumc_to_y, dtype=tf.float32)
        
        # one iteration       
        xe_v_sumc, quantizer_index, syndrome_weight, xe_c_sumv_nonq, xe_c_sumv = self.one_bp_iteration(xe_c2v_pre_iter, xe_0, llr_into_bp_net, H_sumC_to_V, h_minC_to_V, h_min_segmentIDs, H_sumV_to_C, H_xe_v_sumc_to_y)
        
        # start the next iteration
        start_next_iteration = xe_c2v_pre_iter.assign(xe_v_sumc)
        
        # get the final marginal probability and decoded results
        bp_out_llr = tf.add(llr_into_bp_net, tf.matmul(H_xe_v_sumc_to_y, xe_v_sumc))
        dec_out = tf.transpose(tf.floordiv(1-tf.sign(bp_out_llr), 2))
        
        return llr_into_bp_net, xe_c2v_pre_iter_assign, start_next_iteration, dec_out, quantizer_index, syndrome_weight, xe_c_sumv_nonq, xe_c_sumv

    def quantized_decode(self, llr_in, bp_iter_num):
        self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})
        xe_v_sumc = self.sess.run(self.xe_c2v_pre_iter_assign)
        for iter in range(0, bp_iter_num-1):   
           
           quantizer_index = self.sess.run(self.quantizer_index)
           syndrome_weight = self.sess.run(self.syndrome_weight)
           xe_c_sumv_nonq = self.sess.run(self.xe_c_sumv_nonq)
           xe_c_sumv = self.sess.run(self.xe_c_sumv)
           xe_v_sumc = self.sess.run(self.start_next_iteration) 
        y_dec = self.sess.run(self.dec_out)
        return y_dec
