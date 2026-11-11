import numpy as np
import heapq
from numba import njit, prange
from numba.typed import List
from engine.llvm_taps import *

# rabitq quantisation
@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def rRound(datum, t):
    # Ensure datum is a numpy array for vectorised operations
    scalar = datum.astype(np.float32) * t
    rounded = np.floor(scalar) + 0.5
    return rounded.astype(np.float32)

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def next_critical(o_prime, y_cur, t):

    crits = (y_cur[o_prime!=0] + 0.5) / o_prime[o_prime!=0]
    # Filter for values strictly greater than current_t
    new_crits = crits[crits > t].astype(np.float32)
    
    return new_crits

# we back to paper algorithm
@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def get_rabit(o_prime):

    # Initialise variables
    v_max = 0.0
    t_max = 0.0
    t = 1e-9

    crits = List.empty_list(np.float32)
    heapq.heapify(crits)
    seen = set()
        
    # Initial rounding
    y_cur = rRound(o_prime, t)

    # Update expansion variables
    v_max, t_max = update_variables(y_cur, o_prime, t, v_max)

    # While There are still critical values we have not enumerated  
    while True:
        # Get next critical points
        next_crits = next_critical(o_prime, y_cur, t)
        # If there are critical points, push them into the heap
        if next_crits.size > 0:
            for c in next_crits:
                if c not in seen:
                    heapq.heappush(crits, c)
                    seen.add(c)

        # Exit condition
        if len(crits) == 0:
            break
        
        t = heapq.heappop(crits)
        
        # Compute next rounded vector
        y_cur = rRound(o_prime, t)
        if np.max(np.abs(y_cur)) > 2.0:
            break
        else:
            # Update expansion variables
            v_max, t_max = update_variables(y_cur, o_prime, t, v_max)
        
    # Final result
    y_bar = rRound(o_prime, t_max)
    mag = np.linalg.norm(y_bar)
    yu_bar = y_bar + 1.5  # (3/2) = 1.5
        
    return yu_bar, mag


# rabit sub funcitons
# Update expansion variables
@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def update_variables(y_cur, o_prime, t, v_max):

    # Compute sp = y_cur * o_prime' (dot product)
    sp = np.dot(y_cur, o_prime.T)
    # Compute magnitude of y_cur
    mag = np.linalg.norm(y_cur)
            
    v_cur = sp / mag
                            
    # Avoid division by zero
    if v_cur > v_max:
        v_max = v_cur
        t_max = t

    return v_max, t_max


# rabitq grid version
vals = [0, 1, 2, 3]
MAX_VAL = 3
def get_rabit_grid(vector, codebook):
    normalised_rabbit_codebook = codebook / np.linalg.norm(codebook, axis=1, keepdims=True)
    rabit_id = np.argmax(np.dot(normalised_rabbit_codebook, vector))
    return codebook[rabit_id]


def make_grid(dimensions):
    grids= []
    current = [0 for _ in range(dimensions)]
    
    return np.array(extend_grid(grids, current)) - 1.5
    
def extend_grid(grids, current):
    
    finished = False
    
    
    while not finished:
        grids.append(current)
        
        if np.any(current != MAX_VAL):
            current = get_next_grid(current)                
        else:
            finished = True
            
    return grids
        
def get_next_grid(vec):
   index = 0
   finished = False
   
   new_vec = np.copy(vec)
   
   while not finished:
        new_vec[index] += 1
        
        if new_vec[index] == MAX_VAL + 1 and index < len(new_vec):
            new_vec[index] = 0
        else:
            return new_vec
            
        index += 1


# dequantiser
def dequant(query, mag, quantised_data):
    star = query @ quantised_data.T
    return (star - (1.5*np.sum(query)))/ mag

@njit(parallel= False, nogil=True, boundscheck= False)
def unpack_rabit_dequant_bits(query, bitpacked, dims, mag):
    # quant = np.zeros(dims, dtype=np.float32)
    quant = 0
    sigma = np.sum(query)
    
    for i in range(dims):
        which_int = np.uint64(np.floor(np.divide(i, 32)))
        pos_in_int = np.uint64(np.remainder(i, 32))
                
        fetch_bits = (bitpacked[which_int] & ( (3 << 62) >> (2 * pos_in_int)))
        moved_bits = fetch_bits >> (62 - (2 * pos_in_int))

        quant += moved_bits * query[i]

    return (quant - 1.5*sigma) / mag

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_dequant_bits_llvm_og(query, bitpacked, dims, mag):
    quant = 0.0
    sigma = 0.0
    
    # Compute sigma separately
    for i in range(dims):
        sigma += query[i]
    
    inv_sqrt = 1.0 / mag
    
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
    
    return (quant - 1.5*sigma) * inv_sqrt


@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_dequant_bits_llvm(query, bitpacked, dims, mag):
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
def unpack_all_data_rb(query, data, DIMS, mags):

    dat_len = data.shape[0]
    res = np.zeros(dat_len, dtype=np.float32)

    for ind in range(dat_len):
        bitpacked = data[ind]
        mag = mags[ind]
        res[ind] = unpack_dequant_bits_llvm(query, bitpacked, DIMS, mag)

    return res

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def unpack_all_data_rb_single(query, data, DIMS, mags):

    dat_len = 990_000
    res = np.zeros(1, dtype=np.float32)

    bitpacked = data[0]
    mag = mags[0]
    for ind in range(dat_len):        
        res[0] = unpack_dequant_bits_llvm(query, bitpacked, DIMS, mag)

    return res