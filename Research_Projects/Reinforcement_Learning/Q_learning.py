#!/usr/bin/env python3

import numpy as np
import math
import random
#import pickle
#import os

class Agent:
    def __init__(self, n_actions, n_features, gamma):
        # print("initializing python Agent")
        self.n_states = 1
        self.n_actions = 4
        self.learning_rate = 0.01
        self.discount_factor = gamma
        self.q = np.zeros((self.n_states,self.n_actions))
        self.action_probs = np.zeros(self.n_actions)
        random.seed(42)

    def save_model(self):
        pass

    def get_action(self, states):
        # print("getting action")
        actions = np.zeros(len(states),dtype=np.float32)
        for i in range(len(states)):
            state = int(states[i][0])
            beta = 1
            min_prob = 0.01
            for a in range(self.n_actions):
                self.action_probs[a] = math.exp(beta * self.q[state,a])
            denom = sum(self.action_probs)
            min_prob_p = denom * min_prob
            for a in range(self.n_actions):
                if self.action_probs[a] < min_prob_p:
                    self.action_probs[a] = min_prob_p
            denom = sum(self.action_probs)
            self.action_probs /= denom
            # print(self.action_probs)
            rand_action = random.choices([0,1,2,3], weights=self.action_probs, k=1)
            # print(f"before return: {rand_action[0]}")
            actions[i] = rand_action[0]
        # print(actions)
        self.actions = actions
        self.states = states
        return actions

    # def get_action(self, states):
    #     # print("getting action")
    #     state = int(states[0][0])
    #     beta = 1
    #     min_prob = 0.01
    #     for a in range(self.n_actions):
    #         self.action_probs[a] = math.exp(beta * self.q[state,a])
    #     denom = sum(self.action_probs)
    #     min_prob_p = denom * min_prob
    #     for a in range(self.n_actions):
    #         if self.action_probs[a] < min_prob_p:
    #             self.action_probs[a] = min_prob_p
    #     denom = sum(self.action_probs)
    #     self.action_probs /= denom
    #     # print(self.action_probs)
    #     rand_action = random.choices([0,1,2,3], weights=self.action_probs, k=1)
    #     # print(f"before return: {rand_action[0]}")
    #     return float(rand_action[0])

    def train(self, rewards, next_states):
        # print("training")
        # print(transitions)
        for i in range(len(rewards)):
            j = i % len(self.actions)
            old_state = int(self.states[j][0])
            new_state = int(next_states[i][0])
            action = int(self.actions[j])
            reward = rewards[i]
            max_Q = np.max(self.q[new_state,:])
            self.q[old_state,action] += self.learning_rate * (reward + self.discount_factor * max_Q - self.q[old_state,action])

# a = Agent()
# a.train([[0,1,5,0]])
# val = a.get_action([[0]])
# print(f"Test Output:{val}")
