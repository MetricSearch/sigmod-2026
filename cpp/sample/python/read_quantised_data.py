import h5py
import numpy as np

def load_h5(data_path="/Volumes/Data/", dataset="data") -> np.ndarray:
    with h5py.File((data_path),'r') as f:

        data = f[dataset][()]
    return data

def main(rotate, load_path = "/data/kb267/OpenAI3/oai3.h5", store_path="quantisations/openai_rabitq.h5"):
    print(f"Loading")
    data = load_h5(load_path)
    print(f"Loading complete")

    print(data[1])

if __name__ == "__main__":
    rotate = True
    text = ""
    if rotate:
        text = "rotated"
    else:
        text = "no_rotation"

    main(rotate=rotate, load_path=f"../../python/quantisations/openai_rabitq_{text}.h5")