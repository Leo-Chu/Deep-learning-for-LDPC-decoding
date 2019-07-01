import numpy as np
import datetime
import BP_SP_AWGN_Decoder
import BP_MS_AWGN_Decoder
import BP_SP_BSC_Decoder
import BP_MS_BSC_Decoder
import tensorflow as tf
import Modulation
import Transmission
import DataIO
import Quantized_ML_Decoder
import Adaptive_BP_QMS_Decoder

def LDPC_BP_SP_AWGN_test(code, channel, top_config, train_config, simutimes_range, target_err_bits_num, batch_size):
    ## load configurations from top_config
    N = top_config.N_code
    K = top_config.K_code
    H_matrix = code.H_matrix
    SNR_set = top_config.SNR_set
    BP_iter_num = top_config.BP_iter_nums
    function = 'LDPC_BP_SP_AWGN_test'
    
    # build BP decoding network
    bp_decoder = BP_SP_AWGN_Decoder.BP_NetDecoder(H_matrix, batch_size)
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (top_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    # simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (top_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (top_config.decoding_x_file, SNR))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, top_config)
        real_batch_size = batch_size
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = y_receive - s_mod
            ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
            
            
#            x_bits = np.random.randint(0, 2, size=(batch_size, K))
#            u_coded_bits = code.encode_LDPC(x_bits)
#            s_mod = Modulation.BPSK(u_coded_bits)
#            y_receive, ch_noise_sigma, ch_noise = channel.channel_transmit(batch_size, s_mod, SNR)
#            ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
            
            LLR = y_receive * 2.0 / (ch_noise_sigma * ch_noise_sigma)
            ##practical noise
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            #BP decoder
            u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), BP_iter_num[0])
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
#            frame_errs_iter[0] += np.sum(np.sign(np.sum(output_x != x_bits, axis=1)))
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')
    
def LDPC_BP_SP_BSC_test(code, channel, top_config, train_config, simutimes_range, target_err_bits_num, batch_size):
    ## load configurations from top_config
    N = top_config.N_code
    K = top_config.K_code
    H_matrix = code.H_matrix
    crossover_prob_set = top_config.crossover_prob_set
    BP_iter_num = top_config.BP_iter_nums
    function = 'LDPC_BP_SP_BSC_test'
    # build BP decoding network
    bp_decoder = BP_SP_BSC_Decoder.BP_NetDecoder(H_matrix, batch_size)
    
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (top_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    ## simulation starts
    start = datetime.datetime.now()
    for crossover_prob in crossover_prob_set:
        y_recieve_file = format('%s_%.3f.dat' % (top_config.decoding_y_file, crossover_prob))
        x_transmit_file = format('%s_%.3f.dat' % (top_config.decoding_x_file, crossover_prob))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, top_config)
        real_batch_size = batch_size
        
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
             # encode and transmisssion
#            x_bits = np.random.randint(0, 2, size=(batch_size, K))
#            u_coded_bits = code.encode_LDPC(x_bits)
#            s_mod = Modulation.BPSK(u_coded_bits)
#            y_receive, ch_noise = channel.channel_transmit(batch_size, s_mod, crossover_prob)
#            LLR = y_receive * (np.log(1-crossover_prob)-np.log(crossover_prob))
            
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = np.multiply(y_receive,s_mod)
#            ch_noise = y_receive
            LLR = y_receive * (np.log(1-crossover_prob)-np.log(crossover_prob))
            ##practical noise
            practical_crossover_prob = np.mean((1-ch_noise)/2)
            print('Practical Crossover Probability: %.3f' % practical_crossover_prob)
            #BP decoder
            u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), BP_iter_num[0])

            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(crossover_prob) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')

def LDPC_BP_MS_AWGN_test(code, channel, top_config, train_config, simutimes_range, target_err_bits_num, batch_size):
    ## load configurations from top_config
    N = top_config.N_code
    K = top_config.K_code
    H_matrix = code.H_matrix
    SNR_set = top_config.SNR_set
    BP_iter_num = top_config.BP_iter_nums
    alpha = top_config.alpha
    beta = top_config.beta
    function = 'LDPC_BP_MS_AWGN_test'
    # build BP decoding network
    bp_decoder = BP_MS_AWGN_Decoder.BP_NetDecoder(H_matrix, batch_size, alpha, beta)
    
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (top_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (top_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (top_config.decoding_x_file, SNR))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, top_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = y_receive - s_mod
            
            
#            x_bits = np.random.randint(0, 2, size=(batch_size, K))
#            u_coded_bits = code.encode_LDPC(x_bits)
#            s_mod = Modulation.BPSK(u_coded_bits)
#            y_receive, ch_noise_sigma, ch_noise = channel.channel_transmit(batch_size, s_mod, SNR)
            
            LLR = y_receive
            ##practical noise
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), BP_iter_num[0])
            
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')

def LDPC_BP_MS_BSC_test(code, channel, top_config, train_config, simutimes_range, target_err_bits_num, batch_size):
    ## load configurations from top_config
    N = top_config.N_code
    K = top_config.K_code
    H_matrix = code.H_matrix
    crossover_prob_set = top_config.crossover_prob_set
    BP_iter_num = top_config.BP_iter_nums
    alpha = top_config.alpha
    beta = top_config.beta
    function = 'LDPC_BP_MS_BSC_test'
    # build BP decoding network
    bp_decoder = BP_MS_BSC_Decoder.BP_NetDecoder(H_matrix, batch_size, alpha, beta)
    
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (top_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s' % (ber_file, function))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    
    ## simulation starts
    start = datetime.datetime.now()
    for crossover_prob in crossover_prob_set:
        y_recieve_file = format('%s_%.3f.dat' % (top_config.decoding_y_file, crossover_prob))
        x_transmit_file = format('%s_%.3f.dat' % (top_config.decoding_x_file, crossover_prob))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, top_config)
        real_batch_size = batch_size
        
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
             # encode and transmisssion
#            x_bits = np.random.randint(0, 2, size=(batch_size, K))
#            u_coded_bits = code.encode_LDPC(x_bits)
#            s_mod = Modulation.BPSK(u_coded_bits)
#            y_receive, ch_noise = channel.channel_transmit(batch_size, s_mod, crossover_prob)
#            LLR = y_receive
            
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = np.multiply(y_receive,s_mod)
            LLR = y_receive
            ##practical noise
            practical_crossover_prob = np.mean((1-ch_noise)/2)
            print('Practical Crossover Probability: %.3f' % practical_crossover_prob)
            #BP decoder
            u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), BP_iter_num[0])

            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        fout_ber.write(str(crossover_prob) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')
    
def Neural_BP_QMS_AWGN_test(code, channel, top_config, train_config, simutimes_range, target_err_bits_num, batch_size, bp_decoder):
    # load configurations from top_config
    N = top_config.N_code
    K = top_config.K_code
    H_matrix = code.H_matrix
    SNR_set = train_config.SNR_set_for_test
    BP_iter_num = train_config.BP_iter_nums
    alpha = train_config.alpha
    beta = train_config.beta
    function = 'Neural_BP_QMS_AWGN_test'
    
    # build BP decoding network
#    bp_decoder = Quantized_ML_Decoder.BP_NetDecoder(H_matrix, batch_size)

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    # initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (top_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s_%.1f_%.1f' % (ber_file, function, alpha[0], beta[0]))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (top_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (top_config.decoding_x_file, SNR))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, top_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = y_receive - s_mod
            
            
#            x_bits = np.random.randint(0, 2, size=(batch_size, K))
#            u_coded_bits = code.encode_LDPC(x_bits)
#            s_mod = Modulation.BPSK(u_coded_bits)
#            y_receive, ch_noise_sigma, ch_noise = channel.channel_transmit(batch_size, s_mod, SNR)
            
            LLR = y_receive
            ##practical noise
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.quantized_decode(LLR.astype(np.float32), BP_iter_num[0], alpha, beta)
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')
    
def Neural_BP_QMS_BSC_test(code, channel, top_config, train_config, simutimes_range, target_err_bits_num, batch_size):
    ## load configurations from top_config
    N = top_config.N_code
    K = top_config.K_code
    H_matrix = code.H_matrix
    crossover_prob_set = train_config.crossover_prob_set_for_test
    BP_iter_num = train_config.BP_iter_nums
    alpha = train_config.alpha
    beta = train_config.beta
    function = 'Neural_BP_QMS_BSC_test'
    # build BP decoding network
    bp_decoder = Quantized_ML_Decoder.BP_NetDecoder(H_matrix, batch_size)
    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    ## initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (top_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s_%.1f_%.1f' % (ber_file, function, alpha[0], beta[0]))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    ## simulation starts
    start = datetime.datetime.now()
    for crossover_prob in crossover_prob_set:
        y_recieve_file = format('%s_%.3f.dat' % (top_config.decoding_y_file, crossover_prob))
        x_transmit_file = format('%s_%.3f.dat' % (top_config.decoding_x_file, crossover_prob))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, top_config)
        real_batch_size = batch_size
        
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
             # encode and transmisssion
#            x_bits = np.random.randint(0, 2, size=(batch_size, K))
#            u_coded_bits = code.encode_LDPC(x_bits)
#            s_mod = Modulation.BPSK(u_coded_bits)
#            y_receive, ch_noise = channel.channel_transmit(batch_size, s_mod, crossover_prob)
#            LLR = y_receive
            
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = np.multiply(y_receive,s_mod)
            LLR = y_receive
            
            ##practical noise
            practical_crossover_prob = np.mean((1-ch_noise)/2)
            print('Practical Crossover Probability: %.3f' % practical_crossover_prob)
            #BP decoder
            u_BP_decoded = bp_decoder.quantized_decode(LLR.astype(np.float32), BP_iter_num[0], alpha, beta)

            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(crossover_prob) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')

def Adaptive_BP_QMS_AWGN_test(code, channel, top_config, train_config, simutimes_range, target_err_bits_num, batch_size):
    # load configurations from top_config
    N = top_config.N_code
    K = top_config.K_code
    H_matrix = code.H_matrix
    beta_set = top_config.beta_set
    syndrome_weight_determining = top_config.syndrome_weight_determining
    SNR_set = top_config.SNR_set
    BP_iter_num = top_config.BP_iter_nums
    function = 'Adaptive_BP_QMS_AWGN_test'
    
    # build BP decoding network
    bp_decoder = Adaptive_BP_QMS_Decoder.BP_NetDecoder(H_matrix, batch_size, beta_set, syndrome_weight_determining)

    # init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print('Open a tf session!')
    sess.run(init)
    
    # initialize simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times!=0:
        max_batches += 1

    ## generate out ber file
    bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = format('%sBER(%d_%d)_BP(%s)' % (top_config.results_folder, N, K, bp_str))
    ber_file = format('%s_%s_%.1f_%.1f_%.1f' % (ber_file, function, beta_set[0,0], beta_set[0,1], beta_set[0,2]))
    ber_file = format('%s.txt' % ber_file)
    fout_ber = open(ber_file, 'wt')    
    ## simulation starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (top_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (top_config.decoding_x_file, SNR))
        dataio_decode = DataIO.BPdecDataIO(y_recieve_file, x_transmit_file, top_config)
        real_batch_size = batch_size
        # simulation part
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        for ik in range(0, max_batches): 
            print('Batch %d in total %d batches.' % (ik, int(max_batches)), end=' ')
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
                
            #encode and transmisssion
            y_receive, x_bits = dataio_decode.load_next_batch(batch_size, ik)
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            ch_noise = y_receive - s_mod
            
#            x_bits = np.random.randint(0, 2, size=(batch_size, K))
#            u_coded_bits = code.encode_LDPC(x_bits)
#            s_mod = Modulation.BPSK(u_coded_bits)
#            y_receive, ch_noise_sigma, ch_noise = channel.channel_transmit(batch_size, s_mod, SNR)
            
            LLR = y_receive
            ##practical noise
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10*np.log10(1 / (noise_power * 2.0)) 
            print('Practical EbN0: %.2f' % practical_snr)
            
            #BP decoder
            u_BP_decoded = bp_decoder.quantized_decode(LLR.astype(np.float32), BP_iter_num[0])
            #BER
            output_x = code.dec_src_bits(u_BP_decoded)
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num and actual_simutimes >= min_simutimes:
                break
        print('%d bits are simulated!' % (actual_simutimes * K))
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(SNR) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(K * actual_simutimes)
        fout_ber.write(str(ber_iter[0]))
        fout_ber.write('\n')
    #simulation finished
    fout_ber.close()
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    sess.close()
    print('Close the tf session!')
    
def softsign(x_in):
    x_temp = x_in/(np.abs(x_in) + 0.01)
    y_out = np.divide(1-x_temp, 2)
    return y_out

def sigmoid(x_in):
    y_out = 1/(1+np.exp(-x_in))
    return y_out

def Generate_AWGN_Training_Data(code, channel, top_config, train_config, generate_data_for):
    #initialized    
    SNR_set = train_config.SNR_set
    
    if generate_data_for == 'Training':
        batch_size_each_SNR = int(train_config.training_minibatch_size // np.size(train_config.SNR_set))
        total_batches = int(train_config.training_sample_num // train_config.training_minibatch_size)
    elif generate_data_for == 'Test':
        batch_size_each_SNR = int(train_config.test_minibatch_size // np.size(train_config.SNR_set))
        total_batches = int(train_config.test_sample_num // train_config.test_minibatch_size)
    else:
        print('Invalid objective of data generation!')
        exit(0)
    
    ## Data generating starts
    start = datetime.datetime.now()
    if generate_data_for == 'Training':
        fout_feature = open(train_config.training_feature_file, 'wb')
        fout_label = open(train_config.training_label_file, 'wb')
    elif generate_data_for == 'Test':
        fout_feature = open(train_config.test_feature_file, 'wb')
        fout_label = open(train_config.test_label_file, 'wb')
        
    for ik in range(0, total_batches):
        for SNR in SNR_set:
            x_bits, u_coded_bits, s_mod, ch_noise, y_receive = Transmission.AWGN_transmission(SNR, batch_size_each_SNR, top_config, code, channel)
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_feature)  # write features to file
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_label)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")

def Generate_BSC_Training_Data(code, channel, top_config, train_config, generate_data_for):
    #initialized    
    crossover_prob_set = train_config.crossover_prob_set
    
    if generate_data_for == 'Training':
        batch_size_each_crossover_prob = int(train_config.training_minibatch_size // np.size(train_config.crossover_prob_set))
        total_batches = int(train_config.training_sample_num // train_config.training_minibatch_size)
    elif generate_data_for == 'Test':
        batch_size_each_crossover_prob = int(train_config.test_minibatch_size // np.size(train_config.crossover_prob_set))
        total_batches = int(train_config.test_sample_num // train_config.test_minibatch_size)
    else:
        print('Invalid objective of data generation!')
        exit(0)
    
    ## Data generating starts
    start = datetime.datetime.now()
    if generate_data_for == 'Training':
        fout_feature = open(train_config.training_feature_file, 'wb')
        fout_label = open(train_config.training_label_file, 'wb')
    elif generate_data_for == 'Test':
        fout_feature = open(train_config.test_feature_file, 'wb')
        fout_label = open(train_config.test_label_file, 'wb')
        
    for ik in range(0, total_batches):
        for crossover_prob in crossover_prob_set:
            x_bits, u_coded_bits, s_mod, ch_noise, y_receive = Transmission.BSC_transmission(crossover_prob, batch_size_each_crossover_prob, top_config, code, channel)
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_feature)  # write features to file
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_label)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")

def Generate_AWGN_Decoding_Data(top_config, code):
    #initialized    
    SNR_set = top_config.SNR_set
    total_samples = top_config.total_samples
    batch_size = 5000
    K = top_config.K_code
    N = top_config.N_code
    rng = np.random.RandomState(None)
    total_batches = int(total_samples // (batch_size*K))
    ## Data generating starts
    start = datetime.datetime.now()
    for SNR in SNR_set:
        y_recieve_file = format('%s_%.1f.dat' % (top_config.decoding_y_file, SNR))
        x_transmit_file = format('%s_%.1f.dat' % (top_config.decoding_x_file, SNR))
        fout_yrecieve = open(y_recieve_file, 'wb')
        fout_xtransmit = open(x_transmit_file, 'wb')
        for ik in range(0, total_batches):
            x_bits = np.random.randint(0, 2, size=(batch_size, K))
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits)
            noise_awgn = rng.randn(batch_size, N)
            ch_noise_normalize = noise_awgn.astype(np.float32)
            ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
            ch_noise = ch_noise_normalize * ch_noise_sigma
            y_receive = s_mod + ch_noise
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_yrecieve)
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_xtransmit)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")
    
def Generate_BSC_Decoding_Data(top_config, code):
    #initialized    
    crossover_prob_set = top_config.crossover_prob_set
    total_samples = top_config.total_samples
    batch_size = 5000
    K = top_config.K_code
    N = top_config.N_code
    total_batches = int(total_samples // (batch_size*K))
    ## Data generating starts
    start = datetime.datetime.now()
    for crossover_prob in crossover_prob_set:
        y_recieve_file = format('%s_%.3f.dat' % (top_config.decoding_y_file, crossover_prob))
        x_transmit_file = format('%s_%.3f.dat' % (top_config.decoding_x_file, crossover_prob))
        fout_yrecieve = open(y_recieve_file, 'wb')
        fout_xtransmit = open(x_transmit_file, 'wb')
        for ik in range(0, total_batches):
            x_bits = np.random.randint(0, 2, size=(batch_size, K))
            u_coded_bits = code.encode_LDPC(x_bits)
            s_mod = Modulation.BPSK(u_coded_bits) 
            noise_bsc = np.sign(np.random.random(size=(batch_size, N))-crossover_prob)
            y_receive = np.multiply(s_mod,noise_bsc)
            y_receive = y_receive.astype(np.float32)
            y_receive.tofile(fout_yrecieve)  # write features to file
            x_bits = x_bits.astype(np.float32)
            x_bits.tofile(fout_xtransmit)
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)
    print("end\n")