import numpy as np           
import random                
import time                  
import math
import threading
import random

from rollout_thread import RolloutThread
from random import randint
from env.sxsy import SXSY, load_init_map

class Rollout(object):
	
	def __init__(
		self,
		num_task,
		num_episode,
		num_iters,
		map_index,
		use_laser,
		noise_argmax):
		
		self.num_episode = num_episode
		self.num_iters = num_iters

		self.map_index = map_index
		self.init_maps = load_init_map(self.map_index)
		self.use_laser = use_laser
		self.num_task = num_task
		self.noise_argmax = noise_argmax

		self.states, self.tasks, self.actions, self.rewards, self.next_states, self.redundant_steps = [self.holder_factory(self.num_task, self.num_episode) for i in range(6)]

	def _rollout_process(self, task, index, sx, sy, current_policy, num_iters):
		thread_rollout = RolloutThread(
									task = task,
									start_x = sx,
									start_y = sy,
									num_steps = num_iters,
									policy = current_policy,
									map_index = self.map_index,
									use_laser = self.use_laser,
									noise_argmax = self.noise_argmax,
									)

		ep_states, ep_tasks, ep_actions, ep_rewards, ep_next_states, ep_redundant_steps = thread_rollout.rollout()
		
		self.states[task][index] = ep_states
		self.tasks[task][index] = ep_tasks
		self.actions[task][index] = ep_actions
		self.rewards[task][index] = ep_rewards
		self.next_states[task][index] = ep_next_states
		self.redundant_steps[task][index] = ep_redundant_steps

	def holder_factory(self, num_task, num_episode):
		return [ [ [] for j in range(num_episode)] for i in range(num_task) ]

	def rollout_batch(self, current_policy, epoch):
		self.states, self.tasks, self.actions, self.rewards, self.next_states, self.redundant_steps = [self.holder_factory(self.num_task, self.num_episode) for i in range(6)]

		train_threads = []
		
		for task in range(self.num_task):
			for i in range(self.num_episode):
				[sx, sy] = self.init_maps[epoch % len(self.init_maps)][i]
				train_threads.append(threading.Thread(target=self._rollout_process, args=(task, i, sx, sy, current_policy, self.num_iters, )))

		# start each training thread
		for t in train_threads:
			t.start()

		# wait for all threads to finish
		for t in train_threads:
			t.join()		

		return self.states, self.tasks, self.actions, self.rewards, self.next_states, self.redundant_steps