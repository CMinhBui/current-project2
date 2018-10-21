import sys
import numpy as np           		
import random                		
import time                  		
import math
from utils import noise_and_argmax
from env.terrain import Terrain

class RolloutThread(object):
	
	def __init__(
		self,
		task,
		start_x,
		start_y,
		num_steps,
		policy,
		map_index,
		use_laser,
		noise_argmax):
	
		self.task = task
		self.start_x = start_x
		self.start_y = start_y
		self.num_steps = num_steps
		self.policy = policy
		self.noise_argmax = noise_argmax
		self.env = Terrain(map_index, use_laser)

	def rollout(self):
		states, tasks, actions, rewards_of_episode, next_states = [], [], [], [], []
		
		self.env.resetgame(self.task, self.start_x, self.start_y)
		state = self.env.player.getposition()

		step = 1

		while True:
			if step > self.num_steps:
				break

			if self.noise_argmax:
				logit = self.policy[state[0], state[1], self.task, 0]
				action = noise_and_argmax(logit)
			else:
				pi = self.policy[state[0], state[1], self.task, 1]
				action = np.random.choice(range(len(pi)), p = np.array(pi)/ np.sum(pi))  # select action w.r.t the actions prob

			reward, done = self.env.player.action(action)
			
			next_state = self.env.player.getposition()
			
			# Store results
			states.append(state)
			tasks.append(self.task)
			actions.append(action)
			rewards_of_episode.append(reward)

			state = next_state
			next_states.append(next_state)

			if done:   
				break

			step += 1

		redundant_steps = step + self.env.min_dist[self.task][states[-1][1], states[-1][0]] - self.env.min_dist[self.task][self.start_y, self.start_x]
		
		return states, tasks, actions, rewards_of_episode, next_states, redundant_steps