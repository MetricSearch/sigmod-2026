import numpy as np
from numba import njit, types, prange
from engine.llvm_taps import *

@njit(parallel= False, nogil=True, boundscheck= False)
def get_int_value(x_i, dims):
    """
    Quantizes a scalar value to one of 4 levels based on distance.
    Returns an integer index 0, 1, 2, or 3.
    """
    scalar1 = 1.51 / np.sqrt(dims)
    scalar2 = 0.453 / np.sqrt(dims)
    vs = np.array([-scalar1, -scalar2, scalar2, scalar1])
    vdiffs = np.abs(vs - x_i)
    wh = np.argmin(vdiffs)
    return wh

@njit(parallel= False, nogil=True, boundscheck= False)
def quant2mse_rot(X, rot, dims):
    """
    Quantizes the rotated data X.
    X: Input matrix (dims x n)
    rot: Rotation matrix (dims x dims)
    dims: Dimension of the data
    """
    y = X @ rot  # Matrix multiplication
    q2 = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            q2[i, j] = get_int_value(y[i, j], dims)
            
    return q2


@njit(parallel= False, nogil=True, boundscheck= False)
def quant2mse(X, dims):
    """
    Quantizes the rotated data X.
    X: Input matrix (dims x n)
    dims: Dimension of the data
    """
    q2 = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            q2[i, j] = get_int_value(X[i, j], dims)
            
    return q2

@njit(parallel= False, nogil=True, boundscheck= False)
def dequant2mse(X, rot, dims):
    """
    Dequantizes the quantized data X.
    X: Quantized matrix (dims x n) with values 0, 1, 2, 3
    rot: Rotation matrix (dims x dims)
    dims: Dimension of the data
    """
    scalar1 = 1.51 / np.sqrt(dims)
    scalar2 = 0.453 / np.sqrt(dims)
    vs = np.array([-scalar1, -scalar2, scalar2, scalar1])
    
    revs = np.zeros_like(X)
    
    for i in range(4):
        qMap = (X == i)
        revs[qMap] = vs[i]
        
    dq2 = revs @ rot.T  # Inverse rotation
    return dq2

@njit(parallel= False, nogil=True, boundscheck= False)
def unpack_tq_dequant_bits(query, bitpacked, dims, vs):
    quant = 0
    
    for i in range(dims):
        which_int = np.uint64(np.floor(np.divide(i, 32)))
        pos_in_int = np.uint64(np.remainder(i, 32))
                
        fetch_bits = (bitpacked[which_int] & ( (3 << 62) >> (2 * pos_in_int)))
        moved_bits = fetch_bits >> (62 - (2 * pos_in_int))

        quant += vs[moved_bits] * query[i]

    return quant


@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_tq_dequant_bits_llvm(query, bitpacked, dims, vs):

    quant = 0.0
    
    idx = 0
    word = bitpacked[0]
    pos = 0
    
    for i in range(dims):
        # Use our custom LLVM intrinsic
        bits = extract_2bit_fast(word, pos)
        quant += vs[bits] * query[i]
        
        pos += 1
        if pos == 32:
            idx += 1
            pos = 0
            word = bitpacked[idx]
    
    return quant

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_all_data_tq(query, data, DIMS):

    scalar1 = 1.51 / np.sqrt(DIMS)
    scalar2 = 0.453 / np.sqrt(DIMS)
    vs = np.array([-scalar1, -scalar2, scalar2, scalar1])

    dat_len = data.shape[0]
    res = np.zeros(dat_len, dtype=np.float32)

    for ind in range(dat_len):
        bitpacked = data[ind]
        res[ind] = unpack_tq_dequant_bits_llvm(query, bitpacked, DIMS, vs)

    return res


@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_all_data_tq_single(query, data, DIMS):

    scalar1 = 1.51 / np.sqrt(DIMS)
    scalar2 = 0.453 / np.sqrt(DIMS)
    vs = np.array([-scalar1, -scalar2, scalar2, scalar1])

    dat_len = 990_000
    res = np.zeros(1, dtype=np.float32)

    bitpacked = data[0]
    for ind in range(dat_len):
        res[0] = unpack_tq_dequant_bits_llvm(query, bitpacked, DIMS, vs)

    return res