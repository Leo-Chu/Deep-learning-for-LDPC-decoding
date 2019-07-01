import numpy as np

class LDPC:
    def __init__(self, N, K, file_G, file_H):
        self.N = N
        self.K = K
        self.G_matrix, self.H_matrix = self.init_LDPC_G_H(file_G, file_H)

    def init_LDPC_G_H(self, file_G, file_H):
        G_matrix_row_col = np.loadtxt(file_G, dtype=np.int32)
        H_matrix_row_col = np.loadtxt(file_H, dtype=np.int32)
        G_matrix = np.zeros([self.K, self.N], dtype=np.int32)
        H_matrix = np.zeros([self.N-self.K, self.N], dtype=np.int32)
        G_matrix[G_matrix_row_col[:, 0], G_matrix_row_col[:, 1]] = 1
        H_matrix[H_matrix_row_col[:, 0], H_matrix_row_col[:, 1]] = 1
        return G_matrix, H_matrix
    
    def encode_LDPC(self, x_bits):
        u_coded_bits = np.mod(np.matmul(x_bits, self.G_matrix), 2)
#        check = np.mod(np.matmul(u_coded_bits, np.transpose(self.H_matrix)),2)
        return u_coded_bits
        
    def dec_src_bits(self, bp_output):
        return bp_output[:,0:self.K]                 

