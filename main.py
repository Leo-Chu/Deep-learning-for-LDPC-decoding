import Configrations
import numpy as np
import Encode
import Simulation
import Channel
import Quantized_ML_Decoding_Optimization
import Quantized_ML_Decoder
# address configurations
top_config = Configrations.TopConfig()
train_config = Configrations.TrainingConfig(top_config)
code = Encode.LDPC(top_config.N_code, top_config.K_code, top_config.file_G, top_config.file_H)
channel = Channel.AWGN(top_config)
if top_config.function == 'GenAWGNDecData':  
    Simulation.Generate_AWGN_Decoding_Data(top_config, code)

if top_config.function == 'GenBSCDecData':  
    Simulation.Generate_BSC_Decoding_Data(top_config, code)
    
if top_config.function == 'GenAWGNData':    
    channel = Channel.AWGN(top_config)
    # generate training data
    Simulation.Generate_AWGN_Training_Data(code, channel, top_config, train_config, "Training")
    # generate test data
    Simulation.Generate_AWGN_Training_Data(code, channel, top_config, train_config, "Test")

if top_config.function == 'GenBSCData':    
    channel = Channel.BSC(top_config)
    # generate training data
    Simulation.Generate_BSC_Training_Data(code, channel, top_config, train_config, "Training")
    # generate test data
    Simulation.Generate_BSC_Training_Data(code, channel, top_config, train_config, "Test")
    
if top_config.function == 'AWGNdecode':
    channel = Channel.AWGN(top_config)
    batch_size = 1250
    simutimes_range = np.array([np.ceil(1e7 / float(top_config.K_code * batch_size)) * batch_size, np.ceil(1e8 / float(top_config.K_code * batch_size)) * batch_size], np.int32)
    Simulation.LDPC_BP_MS_AWGN_test(code, channel, top_config, train_config, simutimes_range, 1000, batch_size)
    
if top_config.function == 'BSCdecode':
    channel = Channel.BSC(top_config)
    batch_size = 1250
    simutimes_range = np.array([np.ceil(1e7 / float(top_config.K_code * batch_size)) * batch_size, np.ceil(1e8 / float(top_config.K_code * batch_size)) * batch_size], np.int32)
    Simulation.LDPC_BP_SP_BSC_test(code, channel, top_config, train_config, simutimes_range, 1000, batch_size)

if top_config.function == 'AWGNTrain':
    test_iter = 10
    SNR = np.linspace(3.25,3.5,2,dtype=np.float32)
    QMS_training = Quantized_ML_Decoding_Optimization.BP_Training(train_config, top_config, code)
    QMS_training.train_network(top_config.model_id, test_iter, SNR)
        
    
if top_config.function == 'AWGNtest':
    channel = Channel.AWGN(top_config)
    batch_size = 1250
    simutimes_range = np.array([np.ceil(1e7 / float(top_config.K_code * batch_size)) * batch_size, np.ceil(1e8 / float(top_config.K_code * batch_size)) * batch_size], np.int32)
    bp_decoder = Quantized_ML_Decoder.BP_NetDecoder(code.H_matrix, batch_size)
    Simulation.Neural_BP_QMS_AWGN_test(code, channel, top_config, train_config, simutimes_range, 1000, batch_size, bp_decoder)
    
if top_config.function == 'BSCtest':
    channel = Channel.BSC(top_config)
    batch_size = 1250
    simutimes_range = np.array([np.ceil(1e7 / float(top_config.K_code * batch_size)) * batch_size, np.ceil(1e8 / float(top_config.K_code * batch_size)) * batch_size], np.int32)
    Simulation.Neural_BP_QMS_BSC_test(code, channel, top_config, train_config, simutimes_range, 1000, batch_size) 
    
if top_config.function == 'Adaptive Decoder':
    channel = Channel.AWGN(top_config)
    batch_size = 1250
    simutimes_range = np.array([np.ceil(1e7 / float(top_config.K_code * batch_size)) * batch_size, np.ceil(1e8 / float(top_config.K_code * batch_size)) * batch_size], np.int32)
    top_config.beta_set = np.array([[0.7,0.6,0.3]])
    Simulation.Adaptive_BP_QMS_AWGN_test(code, channel, top_config, train_config, simutimes_range, 1000, batch_size)