import time
from numpy.linalg import norm
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from numba import njit, prange


# CSV handling
BASE_DIR = "results/"

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def numba_warm_up():
    np.multiply(np.float32(12), np.float32(15))

def load(f_name):
    return pd.read_csv(BASE_DIR+f_name)

def store(f_name, data):
    f_name = BASE_DIR+f_name
    df = pd.DataFrame(data)
    df.head()
    df.to_csv(
        f_name,
        mode="a",
        header=not Path(f_name).exists(),
        index=False,
    )

def append(df, new_row):
    df.loc[len(df)] = new_row
    return df


def duration(start_time):
    return time.time() - start_time

# data quantisation
def l2_normalise(data):
    if data.ndim > 1:
        normal = norm(data, axis=1, keepdims=True)
    else:
        normal = norm(data)
    return data / normal

def get_dd(data: np.ndarray) -> np.ndarray:
    return (data > 0).astype(np.uint8)

# grond truth handling
def get_ground_truth(data, query, noOfQueries=1000, noOfNNs=100):
    # multiply first 10000 by transpose og 990000
    allSmallSims = query[:noOfQueries].astype(np.float32) @ data.T.astype(np.float32)
    # Argosrt
    gtIds = np.argsort(-allSmallSims,axis=1)[:,:noOfNNs]
    # keep the labels
    return gtIds

def get_recall_at_gt_width(nn_table_pq, ground_knns, sample_size=100, width=100):
    total_overlap = []

    for qid in range(sample_size):
        ground_ids = ground_knns[qid,1:width+1]
        # row_pq = nn_table_pq[qid,:]
        row_pq = nn_table_pq[qid,:width]
        overlap_per_row = len(np.intersect1d(ground_ids, row_pq))
        total_overlap.append(overlap_per_row)
        
    mean_r = np.mean(total_overlap)

    if mean_r < 0:
        return 0
    else:
        return mean_r / (len(row_pq))


def bitpack_q_datum(datum):
    num_ints = np.ceil( (len(datum) // 32), dtype = np.uint64)
    
    ints = np.zeros(num_ints, dtype=np.uint64)
    wh = 0
    
    for elem in datum:
        which_int = wh // 32
        
        ints[which_int] = ints[which_int] << 2
        ints[which_int] = ints[which_int] | elem.astype(np.uint64)
        
        wh += 1

    return ints

def bitpack_q_data(data):
    bitpack_data = []
    for datum in data:
        bitpack_data.append(bitpack_q_datum(datum.astype(np.uint64)))

    return np.array(bitpack_data)


@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def bitpack_q_data_jit(data):
    num_ints = np.uint64(np.ceil( data.shape[1] // 32))
    dat_size = data.shape[0]
    ints = np.zeros((dat_size,num_ints), dtype=np.uint64)
    for ind in range(dat_size):
        datum = data[ind] 
        wh = 0
        for elem in datum:
            which_int = wh // 32
            ints[ind,which_int] = ints[ind,which_int] << 2
            ints[ind,which_int] = ints[ind,which_int] | np.uint64(elem)
            
            wh += 1

    return ints

    
def unpack_bits(bitpacked, dims):
    quant = np.zeros(dims, dtype=np.uint8)
    
    for i in range(dims):
        which_int = np.floor(np.divide(i, 32)).astype(np.uint64)
        pos_in_int = np.remainder(i, 32).astype(np.uint64)
                
        fetch_bits = (bitpacked[which_int] & ( (3 << 62) >> (2 * pos_in_int)))
        moved_bits = fetch_bits >> (62 - (2 * pos_in_int))
        quant[i] = moved_bits
        
    return quant