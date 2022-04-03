#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
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


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

from collections import deque

class Agent:
	def __init__(self, n_actions, n_features, gamma):
		self.n_actions = n_actions
		self.n_features = n_features
		#self.gamma = gamma   #discount factor, need to tune (0.9)
		self.learning_rate = 0.001
		self.epsilon = 1.0
		
		self.epsilon_decay=0.995
	
		self.gamma = 0.95
		#self.tau   = .125

		self.state_buffer = []
		self.state_next_buffer = []
		self.action_buffer = []
		self.reward_buffer = []
		self.max_memory_length = 10000
		self.update_target_frequency = 10

		self.action_values = np.zeros(self.n_actions)
		self.action_probs = np.zeros(self.n_actions)
		self.update_count = 0
		self.action_count = 0
		#actor
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions]) # where we will feed de/dC (from critic)
		
		
		actor_model_weights = self.actor_model.trainable_weights
	
		self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		
		grads = zip(self.actor_grads, actor_model_weights)
		
		self.optimize = tf.optimizers.Adam(self.learning_rate).apply_gradients(grads)


		self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
		
		_, _, self.target_critic_model = self.create_critic_model()

		
		self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input) # where we calcaulte de/dC for feeding above
		
		# Initialize for later gradient calculations
		self.sess = tf.compat.v1.Session()
		self.sess.run(tf.compat.v1.global_variables_initializer())

		
		self.load_weights()
		
		self.load_buffers()
		#self.loss = keras.losses.Huber()
		
		random.seed(42)
		
		
		#self.epsilon = 1.0
		#self.epsilon_min = 0.01
		#self.epsilon_max = 1.0
		#self.epsilon_random_actions = 10000
		#self.epsilon_decay = (self.epsilon_max - self.epsilon_min) / 100000
		

	def create_actor_model(self):
		state_input = Input(shape=(self.n_features))
		h1 = Dense(24, activation='relu')(state_input)
		h2 = Dense(48, activation='relu')(h1)
		h3 = Dense(24, activation='relu')(h2)
		output = Dense(self.n_actions, activation='softmax')(h3)
		
		model = Model(inputs=state_input, outputs=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model
	
	def create_critic_model(self):
		state_input = Input(shape=(self.n_features))
		state_h1 = Dense(24, activation='relu')(state_input)
		state_h2 = Dense(48)(state_h1)
		
		action_input = Input(shape=(self.n_actions))
		action_h1    = Dense(48)(action_input)
		
		merged    = Add()([state_h2, action_h1])
		merged_h1 = Dense(24, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model  = Model(inputs=[state_input,action_input], outputs=output)
		
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	def get_action(self, features):
		actions = self.actor_model.predict(features)
		for i in range(len(actions)):
			if np.random.random() < self.epsilon:
				actions[i] = np.random.choice(self.n_actions)
			self.action_count += 1
		 
		self.epsilon *= self.epsilon_decay

		self.state_buffer.extend(features)
		self.action_buffer.extend(actions)

		return actions

	def _train_actor(self, state_sample,action_sample,reward_sample,state_next_sample):
		for i in range(len(reward_sample)):
			cur_state = state_sample[i]
			action = action_sample[i]
			reward = reward_sample[i]
			new_state = state_next_sample[i]
			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state,
				self.critic_action_input: predicted_action
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state,
				self.actor_critic_grad: grads
			})
			
	def _train_critic(self, state_sample,action_sample,reward_sample,state_next_sample):
		for i in range(len(reward_sample)):
			cur_state = state_sample[i]
			action = action_sample[i]
			reward = reward_sample[i]
			new_state = state_next_sample[i]

			target_action = self.target_actor_model.predict(new_state)
			future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
			reward += self.gamma * future_reward
			self.critic_model.fit([cur_state, action], reward, verbose=0)
		
	def train(self, rewards, next_states):
		self.reward_buffer.extend(rewards)
		self.state_next_buffer.extend(next_states)
		batch_size = 32
		
		if len(self.rewards) < batch_size:
			self.update_count += 1
			return

		indices = np.random.choice(range(len(self.state_buffer)), size=batch_size, replace=False)
		state_sample = np.array([self.state_buffer[i] for i in indices])
		state_next_sample = np.array([self.state_next_buffer[i] for i in indices])
		action_sample = np.array([self.action_buffer[i] for i in indices])
		reward_sample = np.array([self.reward_buffer[i] for i in indices])

		self._train_critic(state_sample,action_sample,reward_sample,state_next_sample)
		self._train_actor(state_sample,action_sample,reward_sample,state_next_sample)

		
		if (self.update_count % self.update_target_frequency == 0):
			self.update_target()

		#limit replay buffer length
		if len(self.state_buffer) > self.max_memory_length:
			del self.state_buffer[:len(self.state_buffer)-self.max_memory_length]
			del self.state_next_buffer[:len(self.state_next_buffer)-self.max_memory_length]
			del self.action_buffer[:len(self.action_buffer)-self.max_memory_length]
			del self.reward_buffer[:len(self.reward_buffer)-self.max_memory_length]

	#def _update_actor_target(self):
	#	actor_model_weights  = self.actor_model.get_weights()
	#	actor_target_weights = self.target_critic_model.get_weights()
	#	
	#	for i in range(len(actor_target_weights)):
	#		actor_target_weights[i] = actor_model_weights[i]
	#	self.target_critic_model.set_weights(actor_target_weights)
#
	#def _update_critic_target(self):
	#	critic_model_weights  = self.critic_model.get_weights()
	#	critic_target_weights = self.critic_target_model.get_weights()
	#	
	#	for i in range(len(critic_target_weights)):
	#		critic_target_weights[i] = critic_model_weights[i]
	#	self.critic_target_model.set_weights(critic_target_weights)		

	def update_target(self):
		self.target_actor_model.set_weights(self.actor_model.get_weights())
		self.target_critic_model.set_weights(self.critic_model.get_weights())
		#self._update_actor_target()
		#self._update_critic_target()

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
		
		

	#input: array of feature vectors of size self.n_features
	#return: array of int32
	#def get_action(self, features): 
	#    action_values = self.model(features, training=False)
#
	#    layer = tf.keras.layers.Softmax()
	#    action_probs = layer(action_values).numpy()
	#    actions = tf.argmax(action_values, axis=1, output_type=dtypes.int32).numpy()
	#    #print(action_probs[0])
	#    #print(probs[0])
	#    for i in range(len(actions)):
	#        actions[i] = np.random.choice(self.n_actions, 1,replace=True, p=action_probs[i])
	#        self.action_count += 1
	#        self.epsilon -= self.epsilon_decay
	#        self.epsilon = max(self.epsilon, self.epsilon_min)
	#    
	#    #actions = tf.argmax(action_values, axis=1, output_type=dtypes.int32).numpy()
	#    #for i in range(len(actions)):
	#    #    if self.action_count < self.epsilon_random_actions or self.epsilon > np.random.rand(1)[0]:
	#    #        actions[i] = np.random.choice(self.n_actions)
	#    #    self.action_count += 1
	#    #    self.epsilon -= self.epsilon_decay
	#    #    self.epsilon = max(self.epsilon, self.epsilon_min)
#
	#    self.state_buffer.extend(features)
	#    self.action_buffer.extend(actions)
	#    # self.state_buffer = np.concatenate((self.state_buffer, features))    #this might be slow
	#    # self.action_buffer = np.concatenate((self.action_buffer, actions))    #this might be slow
#
	#    return actions

	#input: array of transitions (features, action, reward, features')
	#def train(self, rewards, next_states):
	#    self.reward_buffer.extend(rewards)
	#    self.state_next_buffer.extend(next_states)
#
	#    #TODO ensure all new transitions are sampled. Should also include many transitions from history
	#    #TODO check if it helps to break this into smaller batches
	#    indices = np.random.choice(range(len(self.state_buffer)), size=min(len(rewards) * 4,len(self.state_buffer)), replace=False)
#
	#    #Sample each buffer
	#    state_sample = np.array([self.state_buffer[i] for i in indices])
	#    state_next_sample = np.array([self.state_next_buffer[i] for i in indices])
	#    action_sample = np.array([self.action_buffer[i] for i in indices])
	#    reward_sample = np.array([self.reward_buffer[i] for i in indices])
#
	#    #train
	#    future_rewards = self.target(state_next_sample) #Using .predict is MUCH slower (though may help with >>1000 samples)
	#    updated_q_values = reward_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
	#    masks = tf.one_hot(action_sample, self.n_actions)
	#    
	#    with tf.GradientTape() as tape:
	#        q_values = self.model(state_sample)
	#        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
	#        loss = self.loss(updated_q_values, q_action)
	#    
	#    grads = tape.gradient(loss, self.model.trainable_variables)
	#    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
#
	#    self.update_count += 1
	#    if (self.update_count % self.update_target_frequency == 0):
	#        self.target.set_weights(self.model.get_weights())
#
	#    #limit replay buffer length
	#    if len(self.state_buffer) > self.max_memory_length:
	#        del self.state_buffer[:len(self.state_buffer)-self.max_memory_length]
	#        del self.state_next_buffer[:len(self.state_next_buffer)-self.max_memory_length]
	#        del self.action_buffer[:len(self.action_buffer)-self.max_memory_length]
	#        del self.reward_buffer[:len(self.reward_buffer)-self.max_memory_length]
#