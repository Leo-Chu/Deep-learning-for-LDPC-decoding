import numpy as np

# this file defines classes for data io
class TrainingDataIO:
    def __init__(self, feature_filename, label_filename, total_trainig_samples, feature_length, label_length):
        print("Construct the data IO class for training!\n")
        self.fin_feature = open(feature_filename, "rb")
        self.fin_label = open(label_filename, "rb")
        self.total_trainig_samples = total_trainig_samples
        self.feature_length = feature_length
        self.label_length = label_length

    def __del__(self):
        print("Delete the data IO class!\n")
        self.fin_feature.close()
        self.fin_label.close()

    def load_next_mini_batch(self, mini_batch_size, factor_of_start_pos=1):
        # the function is to load the next batch where the datas in the batch are from a continuous memory block
        # the start position for reading data must be a multiple of factor_of_start_pos
        remain_samples = mini_batch_size
        sample_id = np.random.randint(self.total_trainig_samples)   # output a single value which is less than total_trainig_samples
        features = np.zeros((0))
        labels = np.zeros((0))
        if mini_batch_size > self.total_trainig_samples:
            print("Mini batch size should not be larger than total sample size!\n")
        self.fin_feature.seek((self.feature_length * 4) * (sample_id//factor_of_start_pos*factor_of_start_pos), 0)  # float32 = 4 bytes = 32 bits
        self.fin_label.seek((self.label_length * 4) * (sample_id//factor_of_start_pos*factor_of_start_pos), 0)
        
        while 1:
            new_feature = np.fromfile(self.fin_feature, np.float32, self.feature_length * remain_samples)
            new_label = np.fromfile(self.fin_label, np.float32, self.label_length * remain_samples)
            features = np.concatenate((features, new_feature))
            labels = np.concatenate((labels, new_label))
            remain_samples -= len(new_feature) // self.feature_length
            if remain_samples == 0:
                break
            self.fin_feature.seek(0, 0)
            self.fin_label.seek(0, 0)
        features = features.reshape((mini_batch_size, self.feature_length))
        labels = labels.reshape((mini_batch_size, self.label_length))
        return features, labels


class TestDataIO:
    def __init__(self, feature_filename, label_filename, test_sample_num, feature_length, label_length):
        self.fin_feature = open(feature_filename, "rb")
        self.fin_label = open(label_filename, "rb")
        self.test_sample_num = test_sample_num
        self.feature_length = feature_length
        self.label_length = label_length
        self.all_features = np.zeros(0)
        self.all_labels = np.zeros(0)
        self.data_position = 0

    def __del__(self):
        self.fin_feature.close()
        self.fin_label.close()
        
    def seek_file_to_zero(self):  # reset the file pointer to the start of the file
        self.fin_feature.seek(0, 0)
        self.fin_label.seek(0, 0)
        
    def load_batch_for_test(self, batch_size):
        if batch_size > self.test_sample_num:
            print("Batch size should not be larger than total sample size!\n")
        if np.size(self.all_features) == 0:
            self.all_features = np.fromfile(self.fin_feature, np.float32, self.feature_length * self.test_sample_num)
            self.all_labels = np.fromfile(self.fin_label, np.float32, self.label_length * self.test_sample_num)
            self.all_features = np.reshape(self.all_features, [self.test_sample_num, self.feature_length])
            self.all_labels = np.reshape(self.all_labels, [self.test_sample_num, self.label_length])
        features = self.all_features[self.data_position:(self.data_position + batch_size), :]
        labels = self.all_labels[self.data_position:(self.data_position + batch_size), :]
        self.data_position += batch_size
        if self.data_position >= self.test_sample_num:
            self.data_position = 0
        return features, labels
        
class BPdecDataIO:
    def __init__(self, recieve_filename, transmit_filename, top_config):
        print("Construct the data IO class for training!\n")
        self.fin_recieve = open(recieve_filename, "rb")
        self.fin_transmit = open(transmit_filename, "rb")
        self.feature_length = top_config.N_code
        self.label_length = top_config.K_code
        self.N = top_config.N_code
        self.total_samples = top_config.total_samples
        self.start_pos = top_config.start_pos
        
    def __del__(self):
        print("Delete the data IO class!\n")
        self.fin_recieve.close()
        self.fin_transmit.close()

    def load_next_batch(self, batch_size, batch_num):
        # the function is to load the next batch where the datas in the batch are from a continuous memory block
        # the start position for reading data must be a multiple of factor_of_start_pos
        remain_samples = batch_size
        sample_id = batch_num * batch_size + self.start_pos * batch_size   # output a single value which is less than total_trainig_samples

        y_recieve = np.zeros((0))
        x_transmit = np.zeros((0))
        self.fin_recieve.seek((self.feature_length * 4) * sample_id, 0)  # float32 = 4 bytes = 32 bits
        self.fin_transmit.seek((self.label_length * 4) * sample_id, 0)
        while 1:
            new_y_recieve = np.fromfile(self.fin_recieve, np.float32, self.feature_length * remain_samples)
            new_x_transmit = np.fromfile(self.fin_transmit, np.float32, self.label_length * remain_samples)
            y_recieve = np.concatenate((y_recieve, new_y_recieve))
            x_transmit = np.concatenate((x_transmit, new_x_transmit))
            remain_samples -= len(new_y_recieve) // self.feature_length
            if remain_samples == 0:
                break
            self.fin_recieve.seek(0, 0)
            self.fin_transmit.seek(0, 0)
        y_recieve = y_recieve.reshape((batch_size, self.feature_length))
        x_transmit = x_transmit.reshape((batch_size, self.label_length))
        return y_recieve, x_transmit