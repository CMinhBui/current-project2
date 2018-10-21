import numpy as np
from env.map import ENV_MAP
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--map_indexs", nargs="+", type=int, required=True)
args = parser.parse_args()

for map_index in args.map_indexs:
	print("Generate start position for map {}".format(map_index))
	map_array = np.array(ENV_MAP[map_index]['map']).astype(int)

	state_space = [list(z) for z in  zip(np.where(map_array != 0)[1].tolist(), np.where(map_array != 0)[0].tolist())]

	smap = []
	for i in tqdm(range(5000)):
		ep_inits = []
		for e in range(24):
			rands = state_space[np.random.choice(range(len(state_space)))]
			ep_inits.append([rands[0], rands[1]])
		smap.append(ep_inits)

	if not os.path.isdir("env/init_maps"):
		os.mkdir("env/init_maps")
	np.save("env/init_maps/init_map_{}".format(map_index), smap)