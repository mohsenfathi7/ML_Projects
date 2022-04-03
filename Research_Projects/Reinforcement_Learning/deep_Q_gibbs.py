#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import random
import copy
# import cProfile
# import pstats
# import time
import gym
import pandas as pd
import json
from tensorflow.python.framework import dtypes
import pickle
import _pickle as cPickle
import os
import csv
import tensorflow_probability as tfp
class Agent:
	def __init__(self, n_actions, n_features, gamma):
		self.n_actions = n_actions
		self.n_features = n_features
		self.gamma = gamma   #discount factor, need to tune (0.9)
		self.action_values = np.zeros(self.n_actions)
		self.action_probs = np.zeros(self.n_actions)
		self.action_count2 = np.zeros(self.n_actions)
		self.model = self.create_q_model() #TODO load weights
		self.target = self.create_q_model()
		#self.load_weights()
		#self.load_buffers()
		self.optimizer = keras.optimizers.Adam(learning_rate=0.01) #this learning rate is much higher than atari, need to tune
		self.loss = keras.losses.Huber()
		self.state_buffer = []
		self.state_next_buffer = []
		self.action_buffer = []
		self.reward_buffer = []
		
		self.max_memory_length = 100000
		self.update_target_frequency = 1000
		self.update_count = 0
		self.batch_size = 32

		self.action_count = 0
		#self.epsilon = 1.0
		#self.epsilon_min = 0.01
		#self.epsilon_max = 1.0
		#self.epsilon_random_actions = 10000
		#self.epsilon_decay = (self.epsilon_max - self.epsilon_min) / 100000
		self.tau = 10.0
		self.tau_max = 10.0
		self.tau_min = 0.1
		self.tau_decay = (self.tau_max - self.tau_min) / 10000

		random.seed(42)

	def __del__(self):
		print('Destructor called, Employee deleted.')
		self.action_count2 /= self.action_count2.sum()
		print(self.action_count2)
		print(self.tau)


	def create_q_model(self):
		inputs = layers.Input(shape=(self.n_features))
		dense1 = layers.Dense(50, activation="relu")(inputs)
		# dense2 = layers.Dense(50, activation="relu")(dense1)
		action = layers.Dense(self.n_actions, activation="sigmoid")(dense1)  #or maybe softmax?
		return keras.Model(inputs=inputs, outputs=action)

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
		
		

	def gibbs_method(self, tau: float, action_values: list) -> list:
		t = copy.deepcopy(action_values.numpy())
		for i in range(len(action_values)):
			for j in range(4):
				t[i][j] = math.exp(float(t[i][j]/tau))
			s = sum(t[i])
			for j in range(4):
				t[i][j] /= s
		#print(softmax)
		#p = np.array(softmax)
		#p /= p.sum()
		#print(p.sum())
		#print(p)
		#if sum(softmax) != 1.0:
		#	for i in range(len(softmax)):
		#		softmax[i] = softmax[i]*(1./sum(softmax))
		#print(sum(softmax))
		
		return t

	#input: array of feature vectors of size self.n_features
	#return: array of int32
	def get_action(self, features): 
		logits = self.model(features, training=False)
		layer = tf.keras.layers.Softmax()
		#probs = layer(logits).numpy()
		#actions = tf.argmax(logits, axis=1, output_type=dtypes.int32).numpy()
		x = self.gibbs_method(self.tau, logits)
		action_probs = tfp.distributions.Categorical(probs=x)
		temp = action_probs.sample()
		actions = temp.numpy()
		
		#print(x.shape)
		#print("pooria")
		#print(x)
		#tau=100
		for i in range(len(actions)):
			#exp = [math.exp(j / tau) for j in logits[i]]
			#sum_of_exp = sum(exp)
			#soft_max = [j / sum_of_exp for j in exp]

			#x = np.array(soft_max)
			#x /= x.sum()

			#actions[i] = np.random.choice(self.n_actions, p=probs[i])
			self.action_count2[actions[i]] +=1
			self.action_count += 1
			self.tau -= self.tau_decay
			self.tau = max(self.tau_min, self.tau)
		#for i in range(184):
		#	#print(logits[i])
		#	#print("pooria")
			#temp = self.softmax_method(1000, logits[i])
		#	#print(temp)
			#action_probs = tfp.distributions.Categorical(probs=temp)
		#	#print(sum(temp))
			#actions[i] = action_probs.sample()
			#action_probs = np.random.choices([self.n_actions)
			#actions[i] = np.random.choice(self.n_actions, p=temp)
		##action_probs = tfp.distributions.Categorical(probs=probs)
		#temp = action_probs.sample()
		#actions = temp.numpy()
		#layer = tf.keras.layers.Softmax()
		#action_probs = probs.numpy()
		#actions = tf.argmax(action_probs, axis=1, output_type=dtypes.int32).numpy()
		#action_probs = keras_gym.proba_dists.CategoricalDist(logits, boltzmann_tau=1000)
		#temp = action_probs.sample()
		#actions = temp.numpy()
		#print(action_probs[0])
		#print(probs[0])
		#for i in range(len(actions)):
		#	if self.action_count < self.epsilon_random_actions or self.epsilon > np.random.rand(1)[0]:
		#	actions[i] = np.random.choice(self.n_actions, p=action_probs[i])
		#	self.action_count += 1
		#	self.epsilon -= self.epsilon_decay
		#	self.epsilon = max(self.epsilon, self.epsilon_min)

		self.state_buffer.extend(features)
		self.action_buffer.extend(actions)

		#print(actions.shape)
		return actions

	#input: array of transitions (features, action, reward, features')
	def train(self, rewards, next_states):
		self.reward_buffer.extend(rewards)
		self.state_next_buffer.extend(next_states)

		#TODO ensure all new transitions are sampled. Should also include many transitions from history
		#TODO check if it helps to break this into smaller batches
		indices = np.random.choice(range(len(self.state_buffer)), size=min(len(rewards) * 4,len(self.state_buffer)), replace=False)

		#Sample each buffer
		state_sample = np.array([self.state_buffer[i] for i in indices])
		state_next_sample = np.array([self.state_next_buffer[i] for i in indices])
		action_sample = np.array([self.action_buffer[i] for i in indices])
		reward_sample = np.array([self.reward_buffer[i] for i in indices])

		#train
		future_rewards = self.target(state_next_sample) #Using .predict is MUCH slower (though may help with >>1000 samples)
		updated_q_values = reward_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
		masks = tf.one_hot(action_sample, self.n_actions)
		
		with tf.GradientTape() as tape:
			q_values = self.model(state_sample)
			q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
			loss = self.loss(updated_q_values, q_action)
		
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

		self.update_count += 1
		if (self.update_count % self.update_target_frequency == 0):
			self.target.set_weights(self.model.get_weights())

		#limit replay buffer length
		if len(self.state_buffer) > self.max_memory_length:
			del self.state_buffer[:len(self.state_buffer)-self.max_memory_length]
			del self.state_next_buffer[:len(self.state_next_buffer)-self.max_memory_length]
			del self.action_buffer[:len(self.action_buffer)-self.max_memory_length]
			del self.reward_buffer[:len(self.reward_buffer)-self.max_memory_length]