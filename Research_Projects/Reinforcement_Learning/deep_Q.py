#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
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

class Agent:
    def __init__(self, n_actions, n_features, gamma):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = gamma   #discount factor, need to tune (0.9)
        self.alpha = 0.0001
        self.action_values = np.zeros(self.n_actions)
        self.action_count2 = np.zeros(self.n_actions)
        
        self.state_next_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.state_buffer = []
        
        #pooria
        self.action_probs = np.zeros(self.n_actions)
        self.model = self.create_q_model() #TODO load weights
        self.target = self.create_q_model()
        self.load_weights()
        self.load_buffers()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.alpha) #this learning rate is much higher than atari, need to tune
        self.loss = keras.losses.Huber()
        
        
        # self.state_buffer = np.empty((0,self.n_features))
        # self.state_next_buffer = np.empty((0,self.n_features))
        # self.action_buffer = np.empty((0))
        # self.reward_buffer = np.empty((0))
        self.max_memory_length = 10000
        self.update_target_frequency = 1000
        self.update_count = 0
        self.batch_size = 32

        self.action_count = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_max = 1.0
        self.epsilon_random_actions = 10000
        self.epsilon_decay = (self.epsilon_max - self.epsilon_min) / 100000
        random.seed(42)

        #temp start profiling
        # self.pr = cProfile.Profile()
        # self.pr.enable()

    #def save_model(self):
        #pass
        # self.pr.disable()
        # self.pr.dump_stats("python_profile")
        # p = pstats.Stats("python_profile")
        # p.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)
        #self.save_weights()
    def __del__(self):
        print('Destructor called, Employee deleted.')
        self.action_count2 /= self.action_count2.sum()
        print(self.action_count2)

    def create_q_model(self):
        inputs = layers.Input(shape=(self.n_features))
        dense1 = layers.Dense(50, activation="relu")(inputs)
        # dense2 = layers.Dense(50, activation="relu")(dense1)
        action = layers.Dense(self.n_actions, activation="linear")(dense1)  #or maybe softmax?
        return keras.Model(inputs=inputs, outputs=action)

    def save_weights(self):
        self.model.save_weights("/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/deep_Q_weights/weights")
        
        
    def load_weights(self):
        if(os.path.isdir('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/deep_Q_weights')):
            self.model.load_weights("/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/deep_Q_weights/weights")
            #self.target.set_weights(self.model.get_weights())

    def save_buffers(self):
        #df = pd.DataFrame(list(zip(self.state_buffer, self.action_buffer, self.reward_buffer, self.state_next_buffer)), columns =['State', 'Action', 'Reward', 'Next-State'])
        #df.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/buffers.csv', index=False)
        
        
        if len(self.state_buffer) > len(self.state_next_buffer):
            del self.state_buffer[-184:]
            del self.action_buffer[-184:]
        #df = pd.DataFrame((self.state_buffer))
        df = pd.DataFrame(self.state_buffer)
        df.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/state_buffer.csv', index=False, header=False)
        df2 = pd.DataFrame(self.state_next_buffer)
        df2.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/state_next_buffer.csv', index=False, header=False)
        df3 = pd.DataFrame(self.reward_buffer)
        df3.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/reward_buffer.csv', index=False, header=False)
        df4 = pd.DataFrame(self.action_buffer)
        df4.to_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/action_buffer.csv', index=False, header=False)

    def load_buffers(self):
        
        #if(os.path.isfile('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/buffers.csv')):
        #   df = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp2/buffers.csv', header=0)
        ##
        ##    #data_list=df.T.values.tolist()
        #   self.state_buffer=df['State'].tolist()
        #   self.action_buffer=df['Action'].tolist()
        #   self.reward_buffer=df['Reward'].tolist()
        #   self.state_next_buffer=df['Next-State'].tolist()
        
        
        if(os.path.isfile('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/state_buffer.csv')):
            df = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/state_buffer.csv', header=None)       
            temp = df.values
            self.state_buffer.extend(temp)
     
                 
            df2 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/state_next_buffer.csv', header=None)
            temp = df2.values
            self.state_next_buffer.extend(temp)
            
            df3 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/reward_buffer.csv', header=None)
            temp = df3.values
            temp = np.squeeze(temp)
            self.reward_buffer.extend(temp)
        
            df4 = pd.read_csv('/socs/islab_archives/STUDENT_THESIS_dr/POORIA_ISMAILI_MSC_dr/experiments/temp/action_buffer.csv', header=None)
            temp = df3.values
            temp = np.squeeze(temp)
            self.action_buffer.extend(temp)

    #input: array of feature vectors of size self.n_features
    #return: array of int32
    def get_action(self, features): 
        action_values = self.model(features, training=False)
        #print(action_values)
        actions = tf.argmax(action_values, axis=1, output_type=dtypes.int32).numpy()
        for i in range(len(actions)):
            if self.action_count < self.epsilon_random_actions or self.epsilon > np.random.rand(1)[0]:
                actions[i] = np.random.choice(self.n_actions)
            self.action_count2[actions[i]] +=1
            self.action_count += 1
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.state_buffer.extend(features)
        self.action_buffer.extend(actions)
        # self.state_buffer = np.concatenate((self.state_buffer, features))    #this might be slow
        # self.action_buffer = np.concatenate((self.action_buffer, actions))    #this might be slow
        #print(actions.shape)
        #print(self.action_count2)
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
            #print(updated_q_values.shape)
            loss = self.loss(updated_q_values, q_action)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        #print(grads)
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