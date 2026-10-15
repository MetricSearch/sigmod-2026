

import os
import sys
import h5py
import numpy as np
from scipy.stats import special_ortho_group
from tqdm import tqdm
from numba import njit, prange, types
from numba.typed import List
import numpy.typing as npt


sys.path.append(os.path.join(os.path.dirname(__file__), "../src/engine"))

from turboquant import quant2mse, dequant2mse
from rabitq import get_rabit, dequant
from evp import get_evp_richard, de_evp
from utils import bitpack_q_datum, unpack_bits

def make_turboquant_data_og(data):        
    np.random.seed(0)  # rng(0) equivalent
    
    dims = data.shape[1]
    
    # Generate random rotations
    rrot = special_ortho_group.rvs(dims).astype(np.float32)
    
    tq2_mse_data = quant2mse(data, rrot, dims)
    
    return tq2_mse_data, rrot


def make_turboquant_data(data):        
    np.random.seed(0)  # rng(0) equivalent
    
    dims = data.shape[1]
    
    tq2_mse_data = quant2mse(data, dims)
    
    return tq2_mse_data, None

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def make_rabit_data(data):
    d_size = data.shape[0]
    quants = np.empty(data.shape, dtype=np.float32)
    mags = np.zeros(d_size, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    for i in range(d_size):
        quant, mag = get_rabit(data[i])
        quants[i] = quant
        mags[i] = mag
        
    return quants, mags

def make_evp_data(data):
    return get_evp_richard(data) + 1


if __name__ == "__main__":
    np.random.seed(1413)  # rng(0) equivalent
    
    with h5py.File("/data/kb267/mf_dino2.h5") as f:
        data = f["embeddings"][:100].astype(np.float32)

    evp_data = make_evp_data(data)
    
    bitpacked = bitpack_q_datum(evp_data[0])
    
    print(bitpacked)
    
    
    unpacked = unpack_bits(bitpacked, 384)
    
    print(evp_data[0])
    print(unpacked)
    