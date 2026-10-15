import numpy as np
from numba import njit, prange
from engine.llvm_taps import *

# EVP in -1,0,1 form
def get_evp_richard(data: np.ndarray) -> np.ndarray:
    median_row = np.median(np.abs(data), axis=1, keepdims=True)
    res = get_evp_helper(data, median_row)
    return res

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def get_evp_helper(data, median_row):
    d_plus_row = (data > median_row).astype(np.uint8)
    d_minus_row = (data < -median_row).astype(np.uint8)
    return d_plus_row-d_minus_row

def de_evp(query, evps):
    star = query @ evps.T
    return (star - np.sum(query)) / np.sqrt(len(query) / 2)

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_evp_dequant_bits_llvm_og(query, bitpacked, dims):
    quant = 0.0
    sigma = 0.0
    
    # Compute sigma separately
    for i in range(dims):
        sigma += query[i]
    
    inv_sqrt = 1.0 / np.sqrt(0.5 * dims)
    
    idx = 0
    word = bitpacked[0]
    pos = 0
    
    for i in range(dims):
        # Use our custom LLVM intrinsic
        bits = extract_2bit_fast(word, pos)
        quant += bits * query[i]
        
        pos += 1
        if pos == 32:
            idx += 1
            pos = 0
            word = bitpacked[idx]
    
    return (quant - sigma) * inv_sqrt

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True, cache=True)
def unpack_evp_dequant_bits_llvm(query, bitpacked, dims):
    quant = 0.0
    
    idx = 0
    word = bitpacked[0]
    pos = 0
    
    for i in range(dims):
        # Use our custom LLVM intrinsic
        bits = extract_2bit_fast(word, pos)
        quant += bits * query[i]
        
        pos += 1
        if pos == 32:
            idx += 1
            pos = 0
            word = bitpacked[idx]
    
    return quant

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_all_data(query, data, DIMS):

    dat_len = data.shape[0]
    res = np.zeros(dat_len, dtype=np.float32)

    for ind in range(dat_len):
        bitpacked = data[ind]
        res[ind] = unpack_evp_dequant_bits_llvm(query, bitpacked, DIMS)

    return res

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_all_data_single(query, data, DIMS):

    dat_len = 990_000
    res = np.zeros(1, dtype=np.float32)

    bitpacked = data[0]
    for ind in range(dat_len):
        res[0] = unpack_evp_dequant_bits_llvm(query, bitpacked, DIMS)

    return res