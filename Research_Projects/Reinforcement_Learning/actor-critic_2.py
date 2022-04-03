#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()
from tensorflow import keras

from tensorflow.keras import layers
# import math
import random
# import cProfile
# import pstats
# import time
import pandas as pd
import json
from tensorflow.python.framework import dtypes
import pickle
import _pickle as cPickle
import os
import csv


from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add, Multiply
from tensorflow.keras.optimizers import Adam
import keras.backend as K

from collections import deque

class Agent:
	def __init__(self, n_actions, n_features, gamma):
		self.n_actions = n_actions
		self.n_features = n_features
		self.x = 1
		#self.gamma = gamma   #discount factor, need to tune (0.9)
		self.alpha = 0.001
		self.gamma = 0.95
		#self.tau   = .125

		self.state_buffer = []
		self.state_next_buffer = []
		self.action_buffer = []
		self.reward_buffer = []
		self.max_memory_length = 10000
		self.update_target_frequency = 10

		#self.action_values = np.zeros(self.n_actions)
		#self.action_probs = np.zeros(self.n_actions)
		self.update_count = 0
		self.action_count = 0
			
		random.seed(42)
		
		self.actor, self.critic, self.policy = self.build_actor_critic_network()
		self.action_space = [i for i in range(self.n_actions)]


	def build_actor_critic_network(self):
		inputs = Input(shape=(self.n_features))
		delta = Input(shape = [1])
		dense1 = Dense(50, activation="relu")(inputs)
		dense2 = Dense(25, activation="relu")(dense1)
		probs = Dense(self.n_actions, activation="softmax")(dense2) 
		values = Dense(1, activation="linear")(dense2)

		def custom_loss(y_true, y_pred):
			out = K.clip(y_pred, 1e-8, 1-1e-8)
			log_lik = y_true * K.log(out)

			return K.sum(-log_lik*delta)
		
		actor = Model(inputs=[inputs, delta], outputs=[probs])
		actor.compile(optimizer=Adam(self.alpha), loss=custom_loss, experimental_run_tf_function=False)

		critic = Model(inputs=[inputs], outputs=[values])
		critic.compile(optimizer=Adam(self.alpha), loss='mean_squared_error', experimental_run_tf_function=False)

		policy = Model(inputs=[inputs], outputs=[probs])

		return actor, critic, policy

	def get_action(self, features):
		probabilities = self.policy(features)
		values = self.critic(features)
		#actions = np.zeros(184)
		x=[0.25,0.25,0.25,0.25]
		actions = tf.argmax(values, axis=1, output_type=dtypes.int32).numpy()
		for i in range(len(actions)):
			actions[i] = np.random.choice(self.n_actions, p=probabilities[i].numpy())
			self.action_count += 1
		self.state_buffer.extend(features)
		self.action_buffer.extend(actions)
		return actions
#
	def train(self, rewards, next_states):
	
		self.reward_buffer.extend(rewards)

		self.state_next_buffer.extend(next_states)

	#	#indices = np.random.choice(range(len(self.state_buffer)), size=min(len(rewards) * 4,len(self.state_buffer)), replace=False)

		#state_sample = np.array([self.state_buffer[i] for i in range(len(self.state_buffer))])
		#state_next_sample = np.array([self.state_next_buffer[i] for i in range(len(self.state_next_buffer))])
		#action_sample = np.array([self.action_buffer[i] for i in range(len(self.action_buffer))])
		#reward_sample = np.array([self.reward_buffer[i] for i in range(len(self.reward_buffer))])
#
		#self.state_buffer = []
		#self.state_next_buffer = []
		#self.action_buffer = []
		#self.reward_buffer = []
#
#
		#critic_value_ = self.critic(state_next_sample)
		#critic_value = self.critic(state_sample)
		##print(critic_value_)
		##print("pooria")
		##print(self.gamma*critic_value_)
		##print("pooria2")
		##print(reward_sample)
		#target = reward_sample + self.gamma * tf.reduce_max(critic_value_, axis=1)
		#delta = target - tf.reduce_max(critic_value, axis=1)
		##print(delta)
		#actions = tf.one_hot(action_sample, self.n_actions)
	#
		#self.actor.fit([state_sample, delta], actions, verbose=0)
		#self.critic.fit(state_sample, target, verbose=0)

	#	#limit replay buffer length
	#	if len(self.state_buffer) > self.max_memory_length:
	#		del self.state_buffer[:len(self.state_buffer)-self.max_memory_length]
	#		del self.state_next_buffer[:len(self.state_next_buffer)-self.max_memory_length]
	#		del self.action_buffer[:len(self.action_buffer)-self.max_memory_length]
	#		del self.reward_buffer[:len(self.reward_buffer)-self.max_memory_length]

	
	def save_weights(self):
		#self.model.save_weights("/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/deep_Q_weights/weights")
		pass
		
	def load_weights(self):
		pass
		#if(os.path.isdir('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/deep_Q_weights')):
		#    self.model.load_weights("/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/deep_Q_weights/weights")
		#self.target.set_weights(self.model.get_weights())

	def save_buffers(self):
		pass
		#df = pd.DataFrame(list(zip(self.state_buffer, self.action_buffer, self.reward_buffer, self.state_next_buffer)), columns =['State', 'Action', 'Reward', 'Next-State'])
		#df.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/buffers.csv', index=False)
		
	def load_buffers(self):
		pass
		#if(os.path.isfile('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/buffers.csv')):
		#    df = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/buffers.csv', header=0)
		#    self.state_buffer=df['State'].tolist()
		#    self.action_buffer=df['Action'].tolist()
		#    self.reward_buffer=df['Reward'].tolist()
		#    self.state_next_buffer=df['Next-State'].tolist()
		
		

