import time
import numpy as np
import pyperf
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/engine"))

from utils import bitpack_q_data_jit, l2_normalise, numba_warm_up

from paper_experiments.quantisation_functions import make_turboquant_data, make_evp_data, make_rabit_data

from turboquant import unpack_all_data_tq, unpack_all_data_tq_single
from rabitq import unpack_all_data_rb, unpack_all_data_rb_single
from evp import unpack_all_data, unpack_all_data_single, unpack_evp_dequant_bits_llvm


def get_next_normed_random_vec( d: int ) -> np.array:
    """
    :param d: the number of dimensions required
    :return: a l2 normed random vector of dimension d and
    """
    return l2_normalise(np.random.normal(0, 1, (d,)))

def create_random_data(size, dims):
    full_data = []
    split_lim = np.floor(size * 0.99).astype(np.int32)

    for i in tqdm(range(size), desc="Generating Random Data"):
        full_data.append(get_next_normed_random_vec(dims))
    
    data = full_data[:split_lim]
    queries = full_data[split_lim:]
    return data, queries


def bencher(loops, query, dp, dims):
    range_it = range(loops)
    t0 = time.perf_counter()

    for _ in range_it:
        unpack_all_data_single(query, dp, dims)
        # unpack_evp_dequant_bits_llvm(query, dp, dims)

    return time.perf_counter() - t0

if __name__ == "__main__":
    numba_warm_up()
    # runner = pyperf.Runner(loops=990_000)
    runner = pyperf.Runner(loops=200)
    np.random.seed(1413)  # rng(0) equivalent
    # dims = [64,128,256]
    dims = 256
    d_size = 1_000_000

    data, queries = create_random_data(d_size,dims)

    data = np.ascontiguousarray(data, dtype=np.float32)
    query = np.ascontiguousarray(queries[0], dtype=np.float32)

    vx = make_evp_data(data)
    dx = bitpack_q_data_jit(vx)
    # dp = dx[0]
    dp = dx[:1]

    # runner.bench_time_func('distance_calc_1v1', bencher, query, dp, dims)
    runner.bench_time_func('distance_calc_1v1', bencher, query, dp, dims, inner_loops=990_000)