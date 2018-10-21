import numpy as np
import tensorflow as tf

from utils import openai_entropy, mse

class Actor():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope('Actor'):
            self.inputs = tf.placeholder(tf.float32, [None, self.state_size])
            self.actions = tf.placeholder(tf.int32, [None, self.action_size])
            self.advantages = tf.placeholder(tf.float32, [None, ])

            self.W_fc1 = self._fc_weight_variable([self.state_size, 256], name = "W_fc1")
            self.b_fc1 = self._fc_bias_variable([256], self.state_size, name = "b_fc1")
            self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

        with tf.variable_scope("Actions"):
            self.W_fc2 = self._fc_weight_variable([256, self.action_size], name = "W_fc2")
            self.b_fc2 = self._fc_bias_variable([self.action_size], 256, name = "b_fc2")

        self.logits = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2

        self.pi = tf.nn.softmax(self.logits)
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
        self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.advantages)

        self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

    def _fc_weight_variable(self, shape, name='W_fc'):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)


class Critic():
    def __init__(self, state_size):
        self.state_size = state_size

        with tf.variable_scope('Critic'):
            self.inputs = tf.placeholder(tf.float32, [None, self.state_size])
            self.returns = tf.placeholder(tf.float32, [None, ])

            self.W_fc1 = self._fc_weight_variable([self.state_size, 256], name = "W_fc1")
            self.b_fc1 = self._fc_bias_variable([256], self.state_size, name = "b_fc1")
            self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

        with tf.variable_scope("Value"):
            self.W_fc2 = self._fc_weight_variable([256, 1], name = "W_fc3")
            self.b_fc2 = self._fc_bias_variable([1], 256, name = "b_fc3")

            self.value = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2
            
        self.value_loss = tf.reduce_mean(mse(tf.squeeze(self.value), self.returns))
   
        self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
            
    def _fc_weight_variable(self, shape, name='W_fc'):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

class NewA2C():

    def __init__(self,
                state_size,
                action_size,
                clip_range=0.2,
                learning_rate=0.005,
                name="A2c"):

        self.state_size = state_size
        self.action_size = action_size
        self.fixed_lr = learning_rate
        self.fixed_clip_range = clip_range

        # Add this placeholder for having this variable in tensorboard
        self.mean_reward = tf.placeholder(tf.float32)
        self.mean_redundant = tf.placeholder(tf.float32)
        self.ratio_ph = tf.placeholder(tf.float32)
        
        with tf.variable_scope(name):
            self.actor = Actor(state_size = self.state_size, action_size = self.action_size)
            self.critic = Critic(state_size = self.state_size)
        
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.clip_range = tf.placeholder(tf.float32, [])

        self.train_opt_value = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.critic.value_loss)

        # Calculate ratio (pi current policy / pi task policy)
        # Task logits is the placeholder for the logits of its original task
        self.task_logits = tf.placeholder(tf.float32, [None, 8])
        self.task_neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.task_logits, labels = self.actor.actions)
        
        self.ratio = tf.exp(self.task_neg_log_prob - self.actor.neg_log_prob)
        
        self.policy_loss1 = self.actor.policy_loss * self.ratio
        self.policy_loss2 = self.actor.policy_loss * tf.clip_by_value(self.ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        self.policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss1, self.policy_loss2))

        self.train_opt_policy = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.policy_loss)

    def learn_actor(self, sess, states, actions, advantages, task_logits):
        
        feed_dict = {
            self.actor.inputs: states,
            self.actor.actions: actions,
            self.actor.advantages: advantages,
            self.task_logits: task_logits,
            self.learning_rate: self.fixed_lr,
            self.clip_range: self.fixed_clip_range
        }

        policy_loss, ratio, _ = sess.run(
            [self.policy_loss, self.ratio, self.train_opt_policy],
            feed_dict=feed_dict
        )

        return policy_loss, ratio

    def learn_critic(self, sess, states, returns):
        feed_dict = {
            self.critic.inputs: states,
            self.critic.returns: returns,
            self.learning_rate: self.fixed_lr,
            self.clip_range: self.fixed_clip_range
        }

        value_loss, _ = sess.run(
            [self.critic.value_loss, self.train_opt_value],
            feed_dict=feed_dict
        )

        return value_loss

    def learn(self, sess, states, actions, returns, advantages, task_logits):
        #Add lr and cliprange decay

        feed_dict = {
            self.actor.inputs: states,
            self.actor.actions: actions,
            self.actor.advantages: advantages,
            self.critic.inputs: states,
            self.critic.returns: returns,
            self.task_logits: task_logits,
            self.learning_rate: self.fixed_lr,
            self.clip_range: self.fixed_clip_range
        }
        
        policy_loss, value_loss, ratio, _ , _ = sess.run(
            [self.policy_loss, 
            self.critic.value_loss,
            self.ratio,
            self.train_opt_policy,
            self.train_opt_value],
            feed_dict = feed_dict
        )

        return policy_loss, value_loss, ratio

# class A2C():
#     def __init__(self, 
#                 name, 
#                 state_size, 
#                 action_size, 
#                 learning_rate = None):

#         self.state_size = state_size
#         self.action_size = action_size

#         # Add this placeholder for having this variable in tensorboard
#         self.mean_reward = tf.placeholder(tf.float32)
#         self.mean_redundant = tf.placeholder(tf.float32)
#         self.ratio = tf.placeholder(tf.float32)
        
#         with tf.variable_scope(name):
#             self.actor = Actor(state_size = self.state_size, action_size = self.action_size)
#             self.critic = Critic(state_size = self.state_size)

#         self.learning_rate = tf.placeholder(tf.float32, [])
#         self.fixed_lr = learning_rate

#         self.train_opt_policy = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.actor.policy_loss)

#         self.train_opt_value = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.critic.value_loss)

#     def learn_actor(self, sess, states, actions, advantages):

#         feed_dict = {
#                         self.actor.inputs: states, 
#                         self.actor.actions: actions, 
#                         self.actor.advantages: advantages,
#                         self.learning_rate: self.fixed_lr
#                     }

#         policy_loss, _, = sess.run(
#                 [self.actor.policy_loss, self.train_opt_policy], 
#                 feed_dict = feed_dict)

#         return policy_loss

#     def learn_critic(self, sess, states, returns):

#         feed_dict = {
#                         self.critic.inputs: states, 
#                         self.critic.returns: returns,
#                         self.learning_rate: self.fixed_lr,
#                     }

#         value_loss, _, = sess.run(
#                 [self.critic.value_loss, self.train_opt_value], 
#                 feed_dict = feed_dict)

#         return value_loss

#     def learn(self, sess, states, actions, returns, advantages):

#         feed_dict = {
#                         self.actor.inputs: states, 
#                         self.critic.inputs: states, 
#                         self.critic.returns: returns,
#                         self.actor.actions: actions, 
#                         self.actor.advantages: advantages,
#                         self.learning_rate: self.fixed_lr,
#                     }

#         policy_loss, value_loss, _, _ = sess.run(
#             [self.actor.policy_loss, self.critic.value_loss, self.train_opt_policy, self.train_opt_value], 
#             feed_dict = feed_dict)

#         return policy_loss, value_loss

if __name__ == '__main__':
    a2c = A2C(100, 8, 0.05, 0.5)