#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
#tf.compat.v1.disable_eager_execution()
import os
import tensorflow.keras as keras
from keras.models import Model
from tensorflow.keras.layers import Dense, Input

from keras.layers.merge import Add, Multiply
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow_probability as tfp

import random
import pandas as pd
from tensorflow.python.framework import dtypes
import pickle
import _pickle as cPickle
import csv

from collections import deque

import tensorflow as tf


class ActorCriticNetwork(keras.Model):
	def __init__(self, n_actions, fc1_dims=50, fc2_dims=25,
			name='actor_critic', chkpt_dir='/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/actor_critic'):
		super(ActorCriticNetwork, self).__init__()
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

		self.fc1 = Dense(self.fc1_dims, activation='relu')
		self.fc2 = Dense(self.fc2_dims, activation='relu')
		self.v = Dense(1, activation=None)
		self.pi = Dense(n_actions, activation='softmax')

	
	def call(self, state):
	
		value = self.fc1(state)
		value = self.fc2(value)

		v = self.v(value)
		pi = self.pi(value)

		return v, pi

class Agent:
	def __init__(self,n_actions=4,n_features=5, gamma=0.99):
		self.gamma = 0.95
		self.alpha = 0.001
		self.n_actions = n_actions
		self.n_features = n_features
		#self.action = np.zeros(184)
		self.action_space = [i for i in range(self.n_actions)]

		self.state_buffer = []
		self.state_next_buffer = []
		self.action_buffer = []
		self.reward_buffer = []

		self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

		self.actor_critic.compile(optimizer=Adam(learning_rate=self.alpha))


	def get_action(self, features):
		values, probs = self.actor_critic.call(features)
		actions = tf.argmax(values, axis=1, output_type=dtypes.int32).numpy()
		for i in range(len(actions)):
			actions[i] = np.random.choice(self.n_actions, 1,replace=True, p=probs[i].numpy())
		
	
		self.state_buffer.extend(features)
		self.action_buffer.extend(actions)
		return actions
	

	def save_models(self):
		print('... saving models ...')
		self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

	def load_models(self):
		print('... loading models ...')
		self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
		

	def train(self, rewards, next_states):
		
		self.reward_buffer.extend(rewards)
		self.state_next_buffer.extend(next_states)

		state_sample = np.array([self.state_buffer[i] for i in range(len(self.state_buffer))])
		state_next_sample = np.array([self.state_next_buffer[i] for i in range(len(self.state_next_buffer))])
		action_sample = np.array([self.action_buffer[i] for i in range(len(self.action_buffer))])
		reward_sample = np.array([self.reward_buffer[i] for i in range(len(self.reward_buffer))])
		
		done = 0
		self.state_buffer = []
		self.state_next_buffer = []
		self.action_buffer = []
		self.reward_buffer = []

		with tf.GradientTape(persistent=True) as tape:
			state_value, probs = self.actor_critic.call(state_sample)
			state_value_, _ = self.actor_critic.call(state_next_sample)
		#	#state_value = tf.squeeze(state_value)
		#	#state_value_ = tf.squeeze(state_value_)

			action_probs = tfp.distributions.Categorical(probs=probs)
			out = K.clip(action_sample, 1e-8, 1-1e-8)

			log_prob = action_probs.log_prob(out)

			delta = reward_sample + self.gamma * tf.reduce_max(state_value_, axis=1)*(1-int(done)) - tf.reduce_max(state_value, axis=1)
			actor_loss = -log_prob*delta
			critic_loss = delta**2
			total_loss = actor_loss + critic_loss
#
		#gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
		#self.actor_critic.optimizer.apply_gradients(zip(
		#	gradient, self.actor_critic.trainable_variables))
#