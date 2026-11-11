import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/engine"))

from paper_experiments.quantisation_functions import make_turboquant_data, make_evp_data, make_rabit_data


from loader import load_dataset
from utils import bitpack_q_data_jit, l2_normalise
from turboquant import unpack_all_data_tq, unpack_all_data_tq_single
from rabitq import unpack_all_data_rb, unpack_all_data_rb_single
from evp import unpack_all_data, unpack_all_data_single

BASE_DIR = "results/"


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

def store_csv(data, f_name = "results.csv"):
    f_name = BASE_DIR + f_name
    df = pd.DataFrame(data)
    df.head()
    df.to_csv(
        f_name,
        mode="a",
        header=not Path(f_name).exists(),
        index=False,
    )

def run_experiment_helper(func, quant_model, mode, repeats, data, query=[], mags=None):
    results = []
    dx = None
    ex = None
    if mode != "quantise":
        dims = data.shape[1] * 32
    else:
        dims = data.shape[1]
    # data = data[:1]
    size = data.shape[0]
    for i in tqdm(range(repeats), desc=f"{mode} {dims} dims {quant_model} experiment"):
        start = perf_counter()
        if mode == "quantise":
            if quant_model == "evp":
                vx = func(data)
            else:
                vx, ex = func(data)
            dx = bitpack_q_data_jit(vx)
        else:
            if quant_model == "rabit":
                vx = func(query, data, dims, mags)
            else:
                vx = func(query, data, dims)
        end = perf_counter() - start
        
        res = {"quant_model":quant_model,"size":size,"dims":dims,"mode":mode,"run_no":i,"time":end}
        results.append(res)

    store_csv(results)
    return dx, ex

def run_suite_helper(data, repeats, query=[]):
    quant_models = ["evp", "tq", "rabit"]
    quantising_models = {"evp":make_evp_data,"tq":make_turboquant_data,"rabit":make_rabit_data}
    dequantising_models = {"evp":unpack_all_data,"tq":unpack_all_data_tq,"rabit":unpack_all_data_rb}
    # dequantising_models = {"evp":unpack_all_data_single,"tq":unpack_all_data_tq_single,"rabit":unpack_all_data_rb_single}
    for qm in quant_models:
            vx, ex = run_experiment_helper(quantising_models[qm], qm, "quantise", repeats, data)
            run_experiment_helper(dequantising_models[qm], qm, "distance_calculation", repeats, vx, query, ex)

if __name__ == "__main__":
    np.random.seed(1413)  # rng(0) equivalent
    # data_sources = {True:["/data/kb267/mf_dino2"], False:["/data/kb267/mf_dino2"]}
    dims = [64,128,256]
    repeats = 5
    d_size = 1_000_000
    
    # for source in data_sources[mode]:
    # data, queries = load_dataset(source)
    for dim in dims:
        data, queries = create_random_data(d_size,dim)
        data = np.ascontiguousarray(data, dtype=np.float32)
        query = np.ascontiguousarray(queries[0], dtype=np.float32)
        run_suite_helper(data, repeats, query=query)