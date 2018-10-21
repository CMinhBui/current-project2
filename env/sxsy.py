import numpy as np
def load_init_map(map_index):
    return np.load("./env/init_maps/init_map_{}.npy".format(map_index)).tolist()

SXSY = {}