from pathlib import Path
# from utils import l2_normalise
from engine.utils import l2_normalise, get_dd
from scipy.io import loadmat
import numpy as np

EMBEDDING_PATH = Path("/data/kb267/mf_dino2")

DATA_SIZE = 990000

def load_embeddings(path):
    # TODO: sort the files as they are not
    mat_files = sorted(
    path.glob("*.mat"),
    key=lambda f: int(f.stem)
    )

    all_vectors = []

    # TODO: Load the matlab files
    for f in mat_files:
        data = loadmat(f)

        embeddings = data["features"]

        all_vectors.append(embeddings)

    all_vectors = np.vstack(all_vectors)

    print(f'Loaded {all_vectors.shape[0]} vectors of {all_vectors.shape[1]} dimensions')
    return all_vectors

def transform_to_tt(data, batch_size=10000):

    shape = data.shape
    n, d = shape

    resBits = np.empty((n,d // 64),dtype=np.uint64)
    resQuants = np.empty((n,d),dtype=np.int8)


    print("Computing sample mean...")

    sample_limit = min(n, batch_size * 5)
    sample = data[:sample_limit]

    sample_mean = np.ascontiguousarray(sample.mean(axis=0), dtype= np.float32)

    max_abs = np.max(np.abs(sample))
    sample_min = np.abs(np.min(sample))
    scale_div = np.float32(max(max_abs, sample_min) / 127.0)

    print("Streaming transform (single pass)...")
    inv_scale = np.float32(1.0 / scale_div)

    for i in range(0,n,batch_size):
        j = min(i + batch_size, n)
        chunk = np.ascontiguousarray(data[i:j], dtype=np.float32)
        centered = chunk - sample_mean
            
        ddevp = get_dd(centered)
        resBits[i:j] = np.packbits(ddevp, axis=1, bitorder='big').view(np.uint64)
        resQuants[i:j] = (chunk * inv_scale).astype(np.int8)

    return resBits, resQuants

def split_dataset(embeddings, vectors = True):
    if vectors:
        return embeddings[:DATA_SIZE, :], embeddings[DATA_SIZE:, :]
    else:
        return embeddings[:DATA_SIZE], embeddings[DATA_SIZE:]

def load_dataset(embedding_pt="/data/kb267/mf_dino2", normalise=True):
    EMBEDDING_PATH= Path(embedding_pt)
    embeddings = load_embeddings(EMBEDDING_PATH)
    if normalise:
        embeddings = l2_normalise(embeddings)
    data, queries = split_dataset(embeddings)
    return data, queries

def tt_loader():
    embeddings = load_embeddings(EMBEDDING_PATH)
    print("TT transofrmation")
    tt_data, tt_quants = transform_to_tt(embeddings)
    tt_data, tt_queries = split_dataset(tt_data)
    tt_quants_data, tt_quants_queries = split_dataset(tt_quants)
    tt_data_nbit_packed = np.ascontiguousarray(tt_data)
    return tt_data_nbit_packed, tt_queries, tt_quants_data, tt_quants_queries


def evp_loader():
    embeddings = load_embeddings(EMBEDDING_PATH)
    print("EVP transofrmation")
    evp = get_evp_binary(embeddings)
    evp_data, evp_queries = split_dataset(evp, True)
    return evp_data, evp_queries
    # return evp

if __name__ == "__main__":
    data, queries = load_dataset()
    print("Split:")
    print(f'data: {data.shape}, queries: {queries.shape}')