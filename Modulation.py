import numpy as np

def BPSK(u_coded_bits):
    s_mod = u_coded_bits * (-2) + 1
    return s_mod
    