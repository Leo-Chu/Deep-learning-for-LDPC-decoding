import numpy as np

## TopConfig defines some top configurations. Other configurations are set based on TopConfig.
class TopConfig:
    def __init__(self):
        #function
#        'GenDecData1'
        self.function = 'test'
        self.channel = 'AWGN'

        # code
        self.N_code = 1998
        self.K_code = 1512
        self.blk_len = self.N_code
        self.file_G = format('./LDPC_matrix/LDPC_gen_mat_%d_%d.txt' % (self.N_code, self.K_code))
        self.file_H = format('./LDPC_matrix/LDPC_chk_mat_%d_%d.txt' % (self.N_code, self.K_code))
        self.D = np.zeros([self.K_code, self.N_code], dtype = np.float64)
        for i in range(0,self.K_code):
            self.D[i,i] = 1
        
        # BP decoding
        self.BP_iter_nums = np.array([10])    # the number of BP iterations
        self.beta_set = np.array([[0.9,0.8,0.7]])
        self.syndrome_weight_determining = np.array([10,30])
#        SNR
        self.SNR_set = np.array([3.25,3.5], np.float32)
#        self.SNR_set = np.linspace(-0.5,1.75,10,dtype=np.float32)
#        self.SNR_set = np.array([0,0.5,1,1.5,2,2.5,3], np.float32)

        #crossover_prob
#        self.crossover_prob_set = np.array([0.024], np.float32)
#        self.crossover_prob_set = np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.01,0.02,0.03], np.float32)
        self.crossover_prob_set = np.linspace(0.06,0.06,1,dtype=np.float32)
        
        # Save
        self.results_folder = "./results/"
    
        # Decoding Data
        self.total_samples = 5e8
        self.start_pos = 0                                #5e9/(batch_size*N)
        self.decoding_data_folder = "./DecodingData/"
#        self.decoding_data_file = format('%sy_recieve%d_%d' % (self.decoding_data_folder, self.N_code, self.K_code))
        self.decoding_y_file = format('%s%s/%d_%d/y_recieve' % (self.decoding_data_folder, self.channel, self.N_code, self.K_code))
        self.decoding_x_file = format('%s%s/%d_%d/x_transmit' % (self.decoding_data_folder, self.channel, self.N_code, self.K_code))
        
class TrainingConfig:
    def __init__(self, top_config):

        # cov^(1/2) file
        self.corr_para = top_config.corr_para
        
        # network parameters
        self.feature_length = top_config.blk_len
        self.label_length = top_config.K_code
        
        # decode
        self.BP_iter_nums = np.array([10])
        
        # SNR
#        self.SNR_set_for_test = np.linspace(-0.5,1.75,10,dtype=np.float32)
        self.SNR_set_for_test = np.array([3.5],dtype=np.float32)
#        self.SNR_set = np.linspace(1,4,10,dtype=np.float32)
        self.SNR_set = np.array([3.5], np.float32)

        
        self.crossover_prob_set_for_test = np.linspace(0.01,0.06,6,dtype=np.float32)
#        self.crossover_prob_set_for_test = np.array([0.01], np.float32)
#        self.crossover_prob_set = np.array([0.006,0.007,0.01,0.02,0.03], np.float32)
        self.crossover_prob_set = np.array([0.01], np.float32)

        #initialized parameter  
        self.para_file = format('%sSNR%.1f_Iter%d.txt' % (top_config.results_folder, self.SNR_set_for_test[0], self.BP_iter_nums[0]))
        training_para = np.loadtxt(self.para_file, np.float32)
        self.alpha = training_para[0,:]
        self.beta = training_para[1,:]
        
#        self.alpha = 1.6*np.ones([10])
#        self.beta = 0.7*np.ones([10])

        # training data information
        self.training_sample_num = 200000    # the number of training samples. It should be a multiple of training_minibatch_size
        # training parameters
        self.epoch_num = 200000  # the number of training iterations.
        self.training_minibatch_size = 100  # one mini-batch contains equal amount of data generated under different CSNR.
        
        # the data folder
        self.training_data_folder = "./TrainingData/"
        self.test_data_folder = "./TestData/"
        
        # the data in the label file is the ground truth.
#        self.training_feature_file = format('%sfeature%d_%d_%d_2.0.dat' % (self.training_data_folder, self.feature_length, self.label_length, self.training_minibatch_size))
        self.training_feature_file = format('%s%s/feature_%d_%d_%d_%.1f.dat' % (self.training_data_folder, top_config.channel, self.feature_length, self.label_length, self.training_minibatch_size,self.SNR_set[0]))
        self.training_label_file = format('%s%s/label_%d_%d_%d_%.1f.dat' % (self.training_data_folder, top_config.channel, self.feature_length, self.label_length, self.training_minibatch_size,self.SNR_set[0]))

        # test data information
        self.test_sample_num = 10500 # it should be a multiple of test_minibatch_size
        self.test_minibatch_size = 100
#        self.test_feature_file = format('%sfeature%d_%d_%d_2.0.dat' % (self.test_data_folder, self.feature_length, self.label_length, self.test_minibatch_size))
        self.test_feature_file = format('%s%s/feature_%d_%d_%d_%.1f.dat' % (self.test_data_folder, top_config.channel, self.feature_length, self.label_length, self.test_minibatch_size,self.SNR_set[0]))
        self.test_label_file = format('%s%s/label_%d_%d_%d_%.1f.dat' % (self.test_data_folder, top_config.channel, self.feature_length, self.label_length, self.test_minibatch_size,self.SNR_set[0]))