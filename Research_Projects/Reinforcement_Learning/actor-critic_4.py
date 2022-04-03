#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.merge import Add, Multiply
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow_probability as tfp

import math
import random
import copy
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




class Agent:
	def __init__(self, n_actions, n_features, gamma):
		self.n_actions = n_actions
		self.n_features = n_features

		self.alpha = 0.0001
		self.gamma = gamma
		self.action_count2 = np.zeros(self.n_actions)

		self.state_buffer = []
		self.state_next_buffer = []
		self.action_buffer = []
		self.reward_buffer = []

		self.optimizer = keras.optimizers.Adam(learning_rate=self.alpha)
		self.loss = keras.losses.Huber()
		self.max_memory_length = 10000
		self.update_target_frequency = 1000
		self.update_count = 0
		self.batch_size = 32

		self.action_count = 0
		self.total = 0

		self.tau = 10.0
		self.tau_max = 10.0
		self.tau_min = 0.1
		self.tau_decay = (self.tau_max - self.tau_min) / 100000

		random.seed(42) 
		
		self.actor_critic, self.pi = self.build_actor_critic_network()
		self.action_space = [i for i in range(self.n_actions)]
		self.load_buffers()
		self.load_weights()
		self.iter_count = 0

	def __del__(self):
		print('Destructor called, Employee deleted.')
		self.action_count2 /= self.action_count2.sum()
		print(self.action_count2)
		print(self.iter_count)
		print(self.total)

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
		#   for i in range(len(softmax)):
		#       softmax[i] = softmax[i]*(1./sum(softmax))
		#print(sum(softmax))
		
		return t

	def build_actor_critic_network(self):
		inputs = Input(shape=(self.n_features))
		dense1 = Dense(50, activation="relu")(inputs)
		#dense2 = Dense(25, activation="relu")(dense1)
		probs = Dense(self.n_actions, activation="softmax")(dense1) 
		value = Dense(1)(dense1)
		logits = Dense(self.n_actions, activation="sigmoid")(dense1)
		actor_critic = keras.Model(inputs=inputs, outputs=[probs,value])
		policy = Model(inputs=inputs, outputs=logits)
		actor_critic.compile(optimizer=Adam(self.alpha))
		return actor_critic, policy

	def get_action(self, features): 
		#probs2,_ = self.actor_critic(features, training=False)
		#print(probs2.shape)
		#probs2 = tf.keras.activations.softmax(probs)
		logits = self.pi(features, training=False)
		#x = self.gibbs_method(self.tau, logits)
		#layer = tf.keras.layers.Softmax()
		#probs = layer(logits).numpy()
		
		action_probs = tfp.distributions.Categorical(logits=logits)
		temp = action_probs.sample()
		actions = temp.numpy()
		
		for i in range(len(actions)):
			self.action_count2[actions[i]] +=1
			self.action_count += 1
			self.tau -= self.tau_decay
			self.tau = max(self.tau_min, self.tau)

		self.state_buffer.extend(features)
		self.action_buffer.extend(actions)
		self.iter_count = self.iter_count +1
		return actions

		


	def train(self, rewards, next_states):
		self.reward_buffer.extend(rewards)
		self.state_next_buffer.extend(next_states)

		for i in range(len(rewards)):
			self.total = self.total + rewards[i]
#
		indices = np.random.choice(range(len(self.state_buffer)), size=min(len(rewards) * 4,len(self.state_buffer)), replace=False)
		
		state_sample = np.array([self.state_buffer[i] for i in indices])
		state_next_sample = np.array([self.state_next_buffer[i] for i in indices])
		action_sample = np.array([self.action_buffer[i] for i in indices])
		reward_sample = np.array([self.reward_buffer[i] for i in indices])
		masks = tf.one_hot(action_sample, self.n_actions)
		with tf.GradientTape() as tape:
			probs, state_value = self.actor_critic(state_sample)
			_, state_value_ = self.actor_critic(state_next_sample)
			state_value = tf.squeeze(state_value)
			state_value_ = tf.squeeze(state_value_)
			probs = probs + 1e-10
			log_prob = tf.math.log(probs)
			q_action = tf.reduce_sum(tf.multiply(log_prob, masks), axis=1)
	
			predicted = reward_sample + self.gamma * state_value_
			critic_loss = self.loss(predicted, state_value)
			delta = predicted - state_value
			actor_loss = -1 * q_action * delta
			j = tf.keras.backend.sum(actor_loss)
			total_loss = j+critic_loss

		grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

		self.state_buffer = []
		self.state_next_buffer = []
		self.action_buffer = []
		self.reward_buffer = []
		##limit replay buffer length
		#if len(self.state_buffer) > self.max_memory_length:
		#    del self.state_buffer[:len(self.state_buffer)-self.max_memory_length]
		#    del self.state_next_buffer[:len(self.state_next_buffer)-self.max_memory_length]
		#    del self.action_buffer[:len(self.action_buffer)-self.max_memory_length]
		#    del self.reward_buffer[:len(self.reward_buffer)-self.max_memory_length]

	
	def save_weights(self):
		#pass
		self.actor_critic.save_weights("/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/deep_Q_weights/weights")
		
		
	def load_weights(self):
		
		if(os.path.isdir('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/deep_Q_weights')):
			self.actor_critic.load_weights("/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/deep_Q_weights/weights")
		#self.target.set_weights(self.actor_critic.get_weights())

	def save_buffers(self):
		pass
		#if len(self.state_buffer) > len(self.state_next_buffer):
		#    x = len(self.state_next_buffer) - len(self.state_buffer)
		#    del self.state_buffer[x:]
		#    del self.action_buffer[x:]
		##df = pd.DataFrame((self.state_buffer))
		#df = pd.DataFrame(self.state_buffer)
		#df.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/state_buffer.csv', index=False, header=False)
		#df2 = pd.DataFrame(self.state_next_buffer)
		#df2.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/state_next_buffer.csv', index=False, header=False)
		#df3 = pd.DataFrame(self.reward_buffer)
		#df3.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/reward_buffer.csv', index=False, header=False)
		#df4 = pd.DataFrame(self.action_buffer)
		#df4.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/action_buffer.csv', index=False, header=False)

		
	def load_buffers(self):
		pass
		#if(os.path.isfile('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/state_buffer.csv')):
			#df = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/ac-3-7/state_buffer.csv', header=None)
			#self.state_buffer=df.values.tolist()
			#print(np.shape(self.state_buffer))
			#df2 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/ac-3-7/state_next_buffer.csv', header=None)
			#self.state_next_buffer=df2.values.tolist()
			#print(type(self.state_next_buffer))
			#print(np.shape(self.state_next_buffer))
			#df3 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/ac-3-7/reward_buffer.csv', header=None)
			#self.reward_buffer=df3.values.tolist()
			#self.reward_buffer=np.squeeze(self.reward_buffer).tolist()
			#print(np.shape(self.reward_buffer))
			#df4 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/ac-3-7/action_buffer.csv', header=None)
			#self.action_buffer=df4.values.tolist()
			#self.action_buffer=np.squeeze(self.action_buffer).tolist()
			#print(type(self.action_buffer))
			#df = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/state_buffer.csv', header=None)
			#temp = df.values
			#self.state_buffer.extend(temp)
			#     
			#df2 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/state_next_buffer.csv', header=None)
			#temp = df2.values
			#self.state_next_buffer.extend(temp)
			#
			#df3 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/reward_buffer.csv', header=None)
			#temp = df3.values
			#temp = np.squeeze(temp)
			#self.reward_buffer.extend(temp)
		#
			#df4 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/action_buffer.csv', header=None)
			#temp = df3.values
			#temp = np.squeeze(temp)
			#self.action_buffer.extend(temp)
		#   self.action_buffer=df['Action'].tolist()
		#   self.reward_buffer=df['Reward'].tolist()
		#   self.state_next_buffer=df['Next-State'].tolist()
		
		

