import numpy as np           		# Handle matrices
import random                		# Handling random number generation
import time                  		# Handling time calculation
import math
import matplotlib.pyplot as plt 	# Display graphs
import os

class PlotFigure(object):
	
	def __init__(self, save_name, env, num_task):
		self.env = env
		self.state_space = env.state_space
		self.num_task = num_task
		self.save_name = save_name.split("/")[1]
		self.save_folder = "plot/{}".format(save_name.split("/")[0])
		if not os.path.isdir("plot"):
			os.mkdir("plot")
		if not os.path.isdir(os.path.join(self.save_folder)):
			os.mkdir(os.path.join(self.save_folder))

	def _plot_point(self, ax, point, angle, length):
		x, y = point

		endy = length * math.sin(math.radians(angle)) + y
		endx = length * math.cos(math.radians(angle)) + x

		ax.plot([x, endx], [y, endy], color = 'blue')

	def _plot_star(self, ax, orig, lengths, max_length=0.5, angles=[270, 225, 180, 135, 90, 45, 0, 315]):
		max_len = max(lengths)
		for i, angle in enumerate(angles):
			self._plot_point(ax, orig, angle, lengths[i]*1.0 / max_len * max_length)

	def plot(self, policy, epoch):
		plt.clf()

		plt.figure()
		for index in range(self.num_task):
			ax = plt.subplot(1, self.num_task, index + 1)
			plt.title(str(epoch))
			for y in range(self.env.map_array.shape[0]):
				for x in range(self.env.map_array.shape[1]):
					if self.env.MAP[y][x] != 0:
						self._plot_star(ax, (x, y), policy[x,y,index, 1])
						plt.plot([x,], [y,], marker='o', markersize=2, color="green")
					else:
						plt.scatter(x, y, marker='x', color="red")

		if not os.path.isdir(os.path.join(self.save_folder, self.save_name)):
			os.mkdir(os.path.join(self.save_folder, self.save_name))

		plt.savefig(os.path.join(self.save_folder, self.save_name, str(epoch) + '.png'), bbox_inches='tight')