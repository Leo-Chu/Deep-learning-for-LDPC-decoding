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
    def __init__(self, H, batch_size):
        #basic para
        self.H = H
        self.batch_size = batch_size
        #check_matrix
        _, self.v_node_num = np.shape(self.H)
        ii, jj = np.nonzero(self.H)
        loc_nzero_row = np.array([ii, jj])
        self.num_all_edges = np.size(loc_nzero_row[1, :])
        
        #quantize
        self.nambda = 1.5
        self.label = np.array([1/3,1])*self.nambda
        
        #get H&P MatrixCV and MatrixVC
        gm = GetMatrixForBPNet(self.H, loc_nzero_row) 
        self.H_sumC_to_V, self.H_minC_to_V, self.h_minC_to_V, self.h_min_segmentIDs, self.map_H_line_to_row = gm.get_Matrix_CV()
        self.H_x_to_xe0, self.H_sumV_to_C, self.H_xe_v_sumc_to_y = gm.get_Matrix_VC()
        
        #LLR
        self.llr_placeholder = tf.placeholder(tf.float32, [batch_size, self.v_node_num])
        
        self.llr_into_bp_net, self.xe_c2v_pre_iter_assign, self.start_next_iteration, self.dec_out, self.alpha, self.beta = self.build_network()
        self.H_llr_in, self.H_xe_0, self.H_out_llr, self.H_dec_out, self.H_alpha, self.H_beta = self.build_neural_network()
        self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_placeholder))  # transpose the llr matrix to adapt to the matrix operation in BP net decoder.
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session() # open a session
        print('Open a tf session!')
        self.sess.run(init)
        
    def __del__(self):
        self.sess.close()
        print('Close a tf session!')
        
    def one_bp_iteration(self, xe_c2v_pre_iter, xe_0, H_sumC_to_V, h_minC_to_V, h_min_segmentIDs, H_sumV_to_C, alpha, beta):
        xe_c_sumv = self.staircase_quantizer(tf.add(xe_0, alpha*tf.matmul(H_sumV_to_C, xe_c2v_pre_iter)),beta)
        xe_sign = tf.to_float(tf.sign(xe_c_sumv))
        xe_sum_log_img = tf.matmul(H_sumC_to_V, tf.multiply(tf.truediv((1 - xe_sign), [2.0]), [3.1415926]))
        xe_sum_log_complex = tf.complex(tf.zeros([self.num_all_edges, self.batch_size]), xe_sum_log_img)
        xe_product = tf.real(tf.exp(xe_sum_log_complex))
        xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        xe_pd_modified = tf.add(xe_product, xe_product_temp)
        xe_v_minc = tf.segment_min(tf.abs(tf.gather(xe_c_sumv, h_minC_to_V)), h_min_segmentIDs)
        xe_v_sumc = tf.multiply(xe_pd_modified, xe_v_minc)
        return xe_v_sumc
    
    def staircase_quantizer(self, message, beta):
        sigma_square = tf.to_double(5*(10**(tf.log(2*beta)/tf.log(3.0)-5)))
        label = tf.to_double(beta*self.label)
        message = tf.to_double(tf.clip_by_value(message, -beta-0.1, beta+0.1))
        a = label[0]*tf.exp(-tf.square(message-label[0])/2/sigma_square) + label[1]*tf.exp(-tf.square(message-label[1])/2/sigma_square) - label[0]*tf.exp(-tf.square(message+label[0])/2/sigma_square) - label[1]*tf.exp(-tf.square(message+label[1])/2/sigma_square)
        b = tf.exp(-tf.square(message-label[0])/2/sigma_square) + tf.exp(-tf.square(message-label[1])/2/sigma_square) + tf.exp(-tf.square(message+label[0])/2/sigma_square) + tf.exp(-tf.square(message+label[1])/2/sigma_square)
        quantized_message = tf.to_float(1/self.nambda*a/b)
        return quantized_message
    
    def one_nn_iteration(self, xe_c2v_pre_iter, xe_0, alpha, beta):
        xe_c_sumv = self.staircase_quantizer(tf.add(tf.matmul(self.H_x_to_xe0, xe_0), alpha*tf.matmul(self.H_sumV_to_C, xe_c2v_pre_iter)), beta)
        xe_sign = tf.to_float(tf.sign(xe_c_sumv))
        xe_sum_log_img = tf.matmul(self.H_sumC_to_V, tf.multiply(tf.truediv((1 - xe_sign), [2.0]), [3.1415926]))
        xe_sum_log_complex = tf.complex(tf.zeros([self.num_all_edges, self.batch_size]), xe_sum_log_img)
        xe_product = tf.real(tf.exp(xe_sum_log_complex))
        xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        xe_pd_modified = tf.add(xe_product, xe_product_temp)
        xe_v_minc = tf.segment_min(tf.abs(tf.gather(xe_c_sumv, self.h_minC_to_V)), self.h_min_segmentIDs)
        xe_v_sumc = tf.multiply(xe_pd_modified, xe_v_minc)
        bp_out_llr = tf.add(xe_0, alpha*tf.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc))
        return bp_out_llr
    
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
        alpha = tf.placeholder(tf.float32, [1])
        beta = tf.placeholder(tf.float32, [1])
        
        # one iteration       
        xe_v_sumc = self.one_bp_iteration(xe_c2v_pre_iter, xe_0, H_sumC_to_V, h_minC_to_V, h_min_segmentIDs, H_sumV_to_C, alpha, beta)
        
        # start the next iteration
        start_next_iteration = xe_c2v_pre_iter.assign(xe_v_sumc)
        
        # get the final marginal probability and decoded results
        bp_out_llr = tf.add(llr_into_bp_net, alpha*tf.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc))
        dec_out = tf.transpose(tf.floordiv(1-tf.sign(bp_out_llr), 2))
        
        return llr_into_bp_net, xe_c2v_pre_iter_assign, start_next_iteration, dec_out, alpha, beta
    
    def build_neural_network(self):
        H_llr_in = tf.placeholder(tf.float32, [self.num_all_edges, self.batch_size])
        H_xe_0 = tf.placeholder(tf.float32, [self.v_node_num, self.batch_size])
        
        # one iteration initialize
        H_alpha = tf.placeholder(tf.float32, [1])
        H_beta = tf.placeholder(tf.float32, [1])
        
        # one iteration       
        H_out_llr = self.one_nn_iteration(H_llr_in, H_xe_0, H_alpha, H_beta)
        H_dec_out = tf.transpose(tf.floordiv(1-tf.sign(H_out_llr), 2))
        return H_llr_in, H_xe_0, H_out_llr, H_dec_out, H_alpha, H_beta
        
    def quantized_decode_before_nn(self, llr_in, bp_iter_num, alpha, beta):
        self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})
        xe_v_sumc = self.sess.run(self.xe_c2v_pre_iter_assign)
        for iter in range(0, bp_iter_num):   
           xe_v_sumc = self.sess.run(self.start_next_iteration, feed_dict={self.alpha: np.array([alpha[iter]]), self.beta: np.array([beta[iter]])}) 
        xe0_into_nn_net = np.transpose(llr_in)
        llr_into_nn_net = xe_v_sumc
        return llr_into_nn_net, xe0_into_nn_net
    
    def quantized_decode(self, llr_in, bp_iter_num, alpha, beta):
        self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})
        xe_v_sumc = self.sess.run(self.xe_c2v_pre_iter_assign)
        for iter in range(0, bp_iter_num-1):   
           xe_v_sumc = self.sess.run(self.start_next_iteration, feed_dict={self.alpha: np.array([alpha[iter]]), self.beta: np.array([beta[iter]])}) 
        y_dec = self.sess.run(self.dec_out,feed_dict={self.alpha: np.array([alpha[bp_iter_num-1]]), self.beta: np.array([beta[bp_iter_num-1]])})
        return y_dec
    
    def one_nn_decode(self, H_llr_in, H_xe_0, H_alpha, H_beta):
        dec_out = self.sess.run(self.H_out_llr, feed_dict={self.H_llr_in: H_llr_in, self.H_xe_0: H_xe_0, self.H_alpha: H_alpha, self.H_beta: H_beta})
        return dec_out