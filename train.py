# import matplotlib
# matplotlib.use("Agg")
import tensorflow as tf
import os 
import time
import argparse
import sys

from datetime import datetime
from network import *
from multitask_policy import MultitaskPolicy

from env.terrain import Terrain

def training(args, share_exp):
	tf.reset_default_graph()
	
	env = Terrain(args.map_index, args.use_laser)
	policies = []
	
	for i in range(args.num_task):

		# policy_i = A2C(
		# 				name 					= 'A2C_' + str(i),
		# 				state_size 				= env.cv_state_onehot.shape[1], 
		# 				action_size				= env.action_size,
		# 				learning_rate			= args.lr
		# 				)
		policy_i = NewA2C(
						name 					= 'A2C_' + str(i),
						state_size 				= env.cv_state_onehot.shape[1], 
						action_size				= env.action_size,
						learning_rate			= args.lr
						)

		policies.append(policy_i)

	variables = tf.trainable_variables()
	print("Initialized networks, with {} trainable weights.".format(len(variables)))
		
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_frac)

	sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	log_folder = 'logs/{}'.format(args.logname)

	suffix = []
	for arg in vars(args):
		exclude = ['num_tests', 'map_index', 'plot_model', 'save_model', 'num_epochs', 'logname']
		if arg in exclude:
			continue

		if arg == 'share_exp':
			if share_exp:
				suffix.append('share_exp')
			continue
			
		boolean = ['share_exp', 'use_laser', 'use_gae', 'immortal', 'noise_argmax']
		if arg in boolean:
			if getattr(args, arg) != 1:
				continue
			else:
				suffix.append(arg)
				continue

		suffix.append(arg + "_" + str(getattr(args, arg)))

	suffix = '-'.join(suffix)

	if not os.path.isdir(log_folder):
		os.mkdir(log_folder)

	print(os.path.join(log_folder, suffix))
	if os.path.isdir(os.path.join(log_folder, suffix)):
		print("Log folder already exists. Continue training ...")
		test_time = len(os.listdir(os.path.join(log_folder, suffix)))
	else:
		os.mkdir(os.path.join(log_folder, suffix))
		test_time = 0
	
	if test_time == 0:
		writer = tf.summary.FileWriter(os.path.join(log_folder, suffix))
	else:
		writer = tf.summary.FileWriter(os.path.join(log_folder, suffix))
	
	test_name =  "map_" + str(args.map_index) + "_test_" + str(test_time)
	tf.summary.scalar(test_name + "/rewards", tf.reduce_mean([policy.mean_reward for policy in policies], 0))
	tf.summary.scalar(test_name + "/redundant_steps", tf.reduce_mean([policy.mean_redundant for policy in policies], 0))
	tf.summary.scalar(test_name + "/ratio", tf.reduce_mean([policy.ratio_ph for policy in policies], 0))

	write_op = tf.summary.merge_all()

	multitask_agent = MultitaskPolicy(
										map_index 			= args.map_index,
										policies 			= policies,
										writer 				= writer,
										write_op 			= write_op,
										num_task 			= args.num_task,
										num_iters 			= args.num_iters,
										num_episode 		= args.num_episode,
										num_epochs			= args.num_epochs,
										gamma 				= 0.99,
										lamb				= 0.96,
										plot_model 			= args.plot_model,
										save_model 			= args.save_model,
										save_name 			= "{}/{}_{}".format(args.logname, test_name, suffix),
										share_exp 			= share_exp,
										use_laser			= args.use_laser,
										use_gae				= args.use_gae,
										noise_argmax		= args.noise_argmax,
									)

	multitask_agent.train(sess, saver)
	sess.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('--logname', nargs='?', type=str, required=True,
						help="Name of the log folder")
	parser.add_argument('--num_tests', nargs='?', type=int, default = 1,
						help='Number of test to run')
	parser.add_argument('--map_index', nargs='?', type=int, default = 4, 
						help='Index of map'),
	parser.add_argument('--num_task', nargs='?', type=int, default = 2, 
    					help='Number of tasks to train on')
	parser.add_argument('--share_exp', nargs='+', type=int, default = [0],
    					help='Whether to turn on sharing samples on training')
	parser.add_argument('--num_episode', nargs='?', type=int, default = 10,
    					help='Number of episodes to sample in each epoch')
	parser.add_argument('--num_iters', nargs='?', type=int, default = 50,
						help='Number of steps to be sampled in each episode')
	parser.add_argument('--lr', nargs='?', type=float, default = 0.005,
						help='Learning rate')
	parser.add_argument('--use_laser', nargs='?', type=int, default = 0,
						help='Whether to use laser as input observation instead of one-hot vector')
	parser.add_argument('--use_gae', nargs='?', type=int, default = 1,
						help='Whether to use generalized advantage estimate')
	parser.add_argument('--num_epochs', nargs='?', type=int, default = 2000,
						help='Number of epochs to train')
	parser.add_argument('--plot_model', nargs='?', type=int, default = 500,
						help='Plot interval')
	parser.add_argument('--noise_argmax', nargs='?', type=int, default = 0,
						help='Whether touse noise argmax in action sampling')
	parser.add_argument('--save_model', nargs='?', type=int, default = 500,
						help='Saving interval')
	parser.add_argument('--gpu_frac', nargs='?', type=float, default = 1.0,
						help='Fraction of gpu usage')
	args = parser.parse_args()

	start = time.time()
	for i in range(args.num_tests):
		for share_exp in args.share_exp:
			training(args, share_exp)

	print("Done in {} hours".format((time.time() - start)/3600))

