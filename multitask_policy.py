import tensorflow as tf      			# Deep Learning library
import numpy as np           			# Handle matrices
import random                			# Handling random number generation
import time                  			# Handling time calculation
import math
import copy
import threading
import os
import sys 

from rollout import Rollout
from env.map import ENV_MAP
from plot_figure import PlotFigure
from collections import deque			# Ordered collection with ends
from env.terrain import Terrain

class MultitaskPolicy(object):

	def __init__(
			self,
			map_index,
			policies,
			writer,
			write_op,
			num_task,
			num_iters,			
			num_episode,
			num_epochs,
			gamma,
			lamb,
			plot_model,
			save_model,
			save_name,
			share_exp,
			use_laser,
			use_gae,
			noise_argmax
			):

		self.map_index = map_index
		self.PGNetwork = policies

		self.writer = writer
		self.write_op = write_op
		self.use_gae = use_gae

		self.num_task = num_task
		self.num_iters = num_iters
		self.num_epochs = num_epochs
		self.num_episode =  num_episode

		self.gamma = gamma
		self.lamb = lamb
		self.save_name = save_name
		self.plot_model = plot_model
		self.save_model = save_model

		self.share_exp = share_exp
		self.noise_argmax = noise_argmax

		self.env = Terrain(self.map_index, use_laser)

		assert self.num_task <= self.env.num_task

		self.plot_figure = PlotFigure(self.save_name, self.env, self.num_task)

		self.rollout = Rollout(num_task = self.num_task, 
							num_episode = self.num_episode, 
							num_iters = self.num_iters, 
							map_index = self.map_index, 
							use_laser = use_laser, 
							noise_argmax = self.noise_argmax,
							)

	def _discount_rewards(self, episode_rewards, episode_nexts, task, current_value):
		discounted_episode_rewards = np.zeros_like(episode_rewards)
		next_value = 0.0
		if episode_rewards[-1] == 1:
			next_value = 0.0
		else:
			next_value = current_value[episode_nexts[-1][0],episode_nexts[-1][1], task]

		for i in reversed(range(len(episode_rewards))):
			next_value = episode_rewards[i] + self.gamma * next_value  
			discounted_episode_rewards[i] = next_value

		return discounted_episode_rewards.tolist()

	def _GAE(self, episode_rewards, episode_states, episode_nexts, task, current_value):
		ep_GAE = np.zeros_like(episode_rewards)
		TD_error = np.zeros_like(episode_rewards)
		lamda=0.96

		next_value = 0.0
		if episode_rewards[-1] == 1:
			next_value = 0.0
		else:
			next_value = current_value[episode_nexts[-1][0],episode_nexts[-1][1], task]

		for i in reversed(range(len(episode_rewards))):
			TD_error[i] = episode_rewards[i]+self.gamma*next_value-current_value[episode_states[i][0],episode_states[i][1], task]
			next_value = current_value[episode_states[i][0],episode_states[i][1], task]

		ep_GAE[len(episode_rewards)-1] = TD_error[len(episode_rewards)-1]
		weight = self.gamma*lamda
		for i in reversed(range(len(episode_rewards)-1)):
			ep_GAE[i] += TD_error[i]+weight*ep_GAE[i+1]

		return ep_GAE.tolist()	

	def _prepare_current_policy(self, sess, epoch):
		current_policy = {}
		
		for task in range(self.num_task):
			for (x, y) in self.env.state_space:
				state_index = self.env.state_to_index[y][x]
				logit, pi = sess.run(
							[self.PGNetwork[task].actor.logits,
							self.PGNetwork[task].actor.pi],
							feed_dict={
								self.PGNetwork[task].actor.inputs: [self.env.cv_state_onehot[state_index]],
							})
		
				current_policy[x,y,task, 0] = logit.ravel().tolist()
				current_policy[x,y,task, 1] = pi.ravel().tolist()
						
		
		if (epoch+1) % self.plot_model == 0:
			self.plot_figure.plot(current_policy, epoch + 1)
						
		return current_policy

	def _prepare_current_values(self, sess, epoch):
		current_values = {}

		for task in range(self.num_task):
			for (x, y) in self.env.state_space:
				state_index = self.env.state_to_index[y][x]
				v = sess.run(
							self.PGNetwork[task].critic.value, 
							feed_dict={
								self.PGNetwork[task].critic.inputs: [self.env.cv_state_onehot[state_index]],
							})
			
				current_values[x,y,task] = v.ravel().tolist()[0]
						
		
		# if (epoch+1) % self.plot_model == 0 or epoch == 0:
		# 	self.plot_figure.plot(current_values, epoch + 1)

		return current_values

	def _prepare_current_neglog(self, sess, epoch):
		current_neglog = {}

		for task in range(self.num_task):
			for (x, y) in self.env.state_space:
				for action in range(self.env.action_size):
					state_index = self.env.state_to_index[y][x]
					neglog = sess.run(
						self.PGNetwork[task].actor.neg_log_prob,
						feed_dict={
							self.PGNetwork[task].actor.inputs: [self.env.cv_state_onehot[state_index]],
							self.PGNetwork[task].actor.actions: [self.env.cv_action_onehot[action]]
						}
					)
					current_neglog[x, y, action, task] = neglog.ravel().tolist()[0]

		return current_neglog

	def _make_batch(self, sess, epoch):

		current_policy = self._prepare_current_policy(sess, epoch)
		current_values = self._prepare_current_values(sess, epoch)

		'''
		states = [
		    task1		[[---episode_1---],...,[---episode_n---]],
		    task2		[[---episode_1---],...,[---episode_n---]],
		   .
		   .
			task_k      [[---episode_1---],...,[---episode_n---]],
		]
		same as actions, tasks, rewards, values, dones
		
		last_values = [
			task1		[---episode_1---, ..., ---episode_n---],
		    task2		[---episode_1---, ..., ---episode_n---],
		   .
		   .
			task_k      [---episode_1---, ..., ---episode_n---],	
		]
		'''
		states, tasks, actions, rewards, next_states, redundant_steps = self.rollout.rollout_batch(current_policy, epoch)

		observations = [[] for i in range(self.num_task)]
		converted_actions = [[] for i in range(self.num_task)]
		logits = [[] for i in range(self.num_task)]

		for task_idx, task_states in enumerate(states):
			for ep_idx, ep_states in enumerate(task_states):
				observations[task_idx] += [self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]]  for s in ep_states]
				converted_actions[task_idx] += [self.env.cv_action_onehot[a] for a in actions[task_idx][ep_idx]]
				logits[task_idx] += [current_policy[s[0], s[1], task_idx, 0] for s in ep_states]


		returns = [[] for i in range(self.num_task)]
		advantages = [[] for i in range(self.num_task)]

		if not self.use_gae:

			for task_idx in range(self.num_task):
				for ep_idx, (ep_rewards, ep_states, ep_next_states) in enumerate(zip(rewards[task_idx], states[task_idx], next_states[task_idx])):
					ep_dones = list(np.zeros_like(ep_rewards))

					ep_returns = self._discount_rewards(ep_rewards, ep_next_states, task_idx, current_values)
					returns[task_idx] += ep_returns

					ep_values = [current_values[s[0], s[1], task_idx] for s in ep_states]

					# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
			    	# rewards = R + yV(s')
					advantages[task_idx] += list((np.array(ep_returns) - np.array(ep_values)).astype(np.float32))
				
				assert len(returns[task_idx]) == len(advantages[task_idx])
		else:

			for task_idx in range(self.num_task):
				for ep_idx, (ep_rewards, ep_states, ep_next_states) in enumerate(zip(rewards[task_idx], states[task_idx], next_states[task_idx])):
					ep_dones = list(np.zeros_like(ep_rewards))
					
					returns[task_idx] += self._discount_rewards(ep_rewards, ep_next_states, task_idx, current_values)
					advantages[task_idx] += self._GAE(ep_rewards, ep_states, ep_next_states, task_idx, current_values)
					
				assert len(returns[task_idx]) == len(advantages[task_idx])

		for task_idx in range(self.num_task):
			states[task_idx] = np.concatenate(states[task_idx])
			actions[task_idx] = np.concatenate(actions[task_idx])
			redundant_steps[task_idx] = np.mean(redundant_steps[task_idx])

		share_observations = [[] for _ in range(self.num_task)]
		share_actions = [[] for _ in range(self.num_task)]
		share_advantages = [[] for _ in range(self.num_task)]
		share_logits = [[] for _ in range(self.num_task)]

		if self.share_exp:
			assert self.num_task > 1

			for task_idx in range(self.num_task):
				for idx, s in enumerate(states[task_idx]):
						
					if self.env.MAP[s[1]][s[0]] == 2:

						act = actions[task_idx][idx]

						# and share with other tasks
						for other_task in range(self.num_task):
							if other_task == task_idx:
								continue

							share_observations[other_task].append(self.env.cv_state_onehot[self.env.state_to_index[s[1]][s[0]]])
							share_actions[other_task].append(self.env.cv_action_onehot[act])
							share_advantages[other_task].append(advantages[task_idx][idx])
							share_logits[other_task].append(current_policy[s[0], s[1], task_idx, 0])

		return observations, converted_actions, returns, advantages, logits, rewards, share_observations, share_actions, share_advantages, share_logits, redundant_steps
		
		
	def train(self, sess, saver):
		total_samples = {}

		for epoch in range(self.num_epochs):
			# sys.stdout.flush()
			
			# ROLLOUT SAMPLE
			#---------------------------------------------------------------------------------------------------------------------#	
			mb_states, mb_actions, mb_returns, mb_advantages, mb_logits, rewards, mbshare_states, mbshare_actions, mbshare_advantages, mbshare_logits, mb_redundant_steps = self._make_batch(sess, epoch)
			#---------------------------------------------------------------------------------------------------------------------#	

			print('epoch {}/{}'.format(epoch, self.num_epochs), end = '\r', flush = True)
			# UPDATE NETWORK
			#---------------------------------------------------------------------------------------------------------------------#	
			sum_dict = {}
			for task_idx in range(self.num_task):
				assert len(mb_states[task_idx]) == len(mb_actions[task_idx]) == len(mb_returns[task_idx]) == len(mb_advantages[task_idx]) == len(mb_logits[task_idx])
				assert len(mbshare_states[task_idx]) == len(mbshare_actions[task_idx]) == len(mbshare_advantages[task_idx]) == len(mbshare_logits[task_idx])

				if not self.share_exp:
					policy_loss, value_loss, ratio = self.PGNetwork[task_idx].learn(sess, 
																				mb_states[task_idx],
																				mb_actions[task_idx],
																				mb_returns[task_idx],
																				mb_advantages[task_idx],
																				mb_logits[task_idx]
																			)
				else:
					value_loss = self.PGNetwork[task_idx].learn_critic(sess,
																		mb_states[task_idx],
																		mb_returns[task_idx])
					
					policy_loss, ratio = self.PGNetwork[task_idx].learn_actor(sess,
																			mb_states[task_idx] + mbshare_states[task_idx],
																			mb_actions[task_idx] + mbshare_actions[task_idx],
																			mb_advantages[task_idx] + mbshare_advantages[task_idx],
																			mb_logits[task_idx] + mbshare_logits[task_idx])

				sum_dict[self.PGNetwork[task_idx].mean_reward] = np.sum(np.concatenate(rewards[task_idx])) / len(rewards[task_idx])
				sum_dict[self.PGNetwork[task_idx].mean_redundant] = mb_redundant_steps[task_idx]
				sum_dict[self.PGNetwork[task_idx].ratio_ph] = np.mean(ratio)
				
				if task_idx not in total_samples:
					total_samples[task_idx] = 0
					
				total_samples[task_idx] += len(list(np.concatenate(rewards[task_idx])))

			#---------------------------------------------------------------------------------------------------------------------#	
			

			# WRITE TF SUMMARIES
			#---------------------------------------------------------------------------------------------------------------------#	
			summary = sess.run(self.write_op, feed_dict = sum_dict)

			self.writer.add_summary(summary, total_samples[0])
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	

			# SAVE MODEL
			#---------------------------------------------------------------------------------------------------------------------#	
			# if epoch % self.save_model == 0:
			# 	saver.save(sess, 'checkpoints/' + self.save_name + '.ckpt')
			#---------------------------------------------------------------------------------------------------------------------#		