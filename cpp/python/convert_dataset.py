from utils.io import load_h5, store_h5
import numpy as np
from tqdm import tqdm
from rabitqlib import Quantiser


def quantise(data: np.ndarray, rotate= False) -> np.ndarray:
    quantised_data = []
    dim = data.shape[1]

    q = Quantiser()
    for datum in tqdm(data):
        median = np.median(datum)
        centroid = np.full(datum.shape, median)
            
        code, _ = q.quantise_single(
            datum,
            centroid,
            dim,
            2,
            rotate
        )

        quantised_data.append(code)

    return quantised_data

def main(rotate, load_path = "/data/kb267/OpenAI3/oai3.h5", store_path="quantisations/openai_rabitq.h5"):
    print(f"Loading")
    data = load_h5(load_path)
    print(f"Loading complete")

    print(f"Quantising")
    quantised_data = quantise(data)
    print(f"Quantisation complete")

    print(f"Storing")
    store_h5(quantised_data, store_path)
    print(f"Storing complete")

if __name__ == "__main__":
    for rotate in [True, False]:
        text = ""
        if rotate:
            text = "rotated"
        else:
            text = "no_rotation"

        main(rotate=rotate, store_path=f"quantisations/openai_rabitq_{text}.h5")