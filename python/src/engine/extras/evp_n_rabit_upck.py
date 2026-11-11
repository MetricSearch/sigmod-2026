from numba import njit, prange
from engine.llvm_taps import *
import numpy as np


@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_dequant_bits_faster(query, bitpacked, dims):
    quant = 0.0
    sigma = np.sum(query)
    
    # Precompute normalisation constant
    inv_sqrt = 1.0 / np.sqrt(0.5 * dims)

    # Manual pos tracking
    idx = 0
    pos = 0
    current_int = bitpacked[0]
    base_shift = 62
    
    for i in range(dims):
        if pos == 32:
            idx += 1
            pos = 0
            current_int = bitpacked[idx]
        
        mask = (3 << 62) >> (2 * pos)
        moved_bits = (current_int & mask) >> (base_shift - 2 * pos)
        
        quant += moved_bits * query[i]
        pos += 1
    
    return (quant - sigma) * inv_sqrt

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_dequant_bits_no_np(query, bitpacked, dims):
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
        # Inline bit extraction (minimal operations)
        bits = (word >> (62 - 2*pos)) & 3
        quant += bits * query[i]
        
        pos += 1
        if pos == 32:
            idx += 1
            pos = 0
            word = bitpacked[idx]
    
    return (quant - sigma) * inv_sqrt


@njit(parallel= False, nogil=True, boundscheck= False)
def unpack_dequant_bits(query, bitpacked, dims):
    # quant = np.zeros(dims, dtype=np.float32)
    quant = 0
    sigma = np.sum(query)
    
    for i in range(dims):
        which_int = np.uint64(np.floor(np.divide(i, 32)))
        pos_in_int = np.uint64(np.remainder(i, 32))
                
        fetch_bits = (bitpacked[which_int] & ( (3 << 62) >> (2 * pos_in_int)))
        moved_bits = fetch_bits >> (62 - (2 * pos_in_int))

        quant += moved_bits * query[i]

    return (quant - sigma) / np.sqrt(dims/ 2)


@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_batch_simd(query, bitpacked, dims):
    """
    Fixed version with proper type handling for float32/64 queries.
    """
    batch_size = bitpacked.shape[0]
    words_per_vector = (dims + 31) // 32
    res = np.empty(batch_size, dtype=np.float64)
    
    inv_sqrt = 1.0 / np.sqrt(0.5 * dims)
    
    # Compute sigma once (works for both float32 and float64)
    sigma = 0.0
    for i in range(dims):
        sigma += query[i]
    
    # Parallel over batch size
    for b in range(batch_size):
        quant = 0.0
        word_idx = 0
        pos = 0
        
        for i in range(dims):
            # Get the current uint64 word
            if pos == 0:
                word = bitpacked[b, word_idx]
            
            # Bit extraction - word is uint64, shift is int64 (compatible!)
            bits = (word >> (62 - 2*pos)) & 3
            
            # Cast bits to float for multiplication
            quant += float(bits) * query[i]
            
            pos += 1
            if pos == 32:
                word_idx += 1
                pos = 0
        
        res[b] = (quant - sigma) * inv_sqrt
    
    return res