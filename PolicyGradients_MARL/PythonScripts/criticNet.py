# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:23:27 2019

@author: jjling.2018
"""
import tensorflow as tf
import numpy as np
import sys
import random
import time
import os

h1 = 64
h2 = 64
h3 = 64
h4 = 64
h5 = 64

class HC_Critic:
	def __init__(self, T_max, num_of_agents, num_of_zones, s, obs, dis_actions, con_actions):
		self.num_of_agents = num_of_agents
		self.num_of_zones = num_of_zones
		self.len_state_single = num_of_zones*2+T_max+1
		self.global_state = s
		self.obs = obs
		self.dis_actions = dis_actions
		self.con_actions = con_actions
		self.gamma = 0.99      
		self.q_value_mix_next = None
		self.r = None         
		self.td_error = None
		self.loss = None
		self.learning_rate = None
		self.optimizer = None
		self.learning_step = None

		with tf.variable_scope('qmix'):   
			self.output_var = self.generate_qmix(scope='eval', trainable=True)
			self.output_var_target = self.generate_qmix(scope='target', trainable=False)
		self.action_gradients = tf.gradients(self.output_var[1], self.con_actions)

	def setCompGraph(self):
		with tf.variable_scope('optimisation'):   
			self.q_value_mix_next = tf.placeholder(shape=[None, 1], dtype=tf.float32)
			self.r = tf.placeholder(shape=[None, 1], dtype=tf.float32)					
			self.td_error = self.r + self.gamma * self.q_value_mix_next - self.output_var[1]
			self.loss = tf.reduce_mean(tf.square(self.td_error))
			
			# Learning Rate
			self.learning_rate = 0.01

			# Defining Optimizer
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

			# Update Gradients
			self.learning_step = (self.optimizer.minimize(self.loss))

	
	def generate_qmix(self, scope, trainable):		
		with tf.variable_scope(scope):			
			q_values_output = {}
			for nn in range(self.num_of_agents):				
				state_single = tf.slice(self.global_state, [0, nn, 0], [-1, 1, self.len_state_single])
				state_single = tf.reshape(tensor=state_single, shape=(-1, self.len_state_single))

				obs_single = tf.slice(self.obs, [0, nn, 0], [-1, 1, self.num_of_zones])
				obs_single = tf.reshape(tensor=obs_single, shape=(-1, self.num_of_zones))

				dis_action_single = tf.slice(self.dis_actions, [0, nn, 0], [-1, 1, 4])
				dis_action_single = tf.reshape(tensor=dis_action_single, shape=(-1, 4))
				
				con_action_single = tf.slice(self.con_actions, [0, nn, 0], [-1, 1, 1])
				con_action_single = tf.reshape(tensor=con_action_single, shape=(-1, 1))

				# generate_single_q_network
				q_hidden_1 = tf.layers.dense(inputs=tf.concat([state_single, obs_single, dis_action_single, con_action_single], 1), units=h1, activation=tf.nn.relu,
										# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
										# bias_initializer=tf.constant_initializer(0.00),  # biases
										use_bias=True,
										trainable=trainable, name='q_dense_h1_agent'+str(nn))
				q_hidden_2 = tf.layers.dense(inputs=q_hidden_1, units=h2, activation=tf.nn.relu,
										# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
										# bias_initializer=tf.constant_initializer(0.00),  # biases
										use_bias=True,
										trainable=trainable, name='q_dense_h2_agent'+str(nn))
				q_hidden_3 = tf.layers.dense(inputs=q_hidden_2, units=h3, activation=tf.nn.relu,
										# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
										# bias_initializer=tf.constant_initializer(0.00),  # biases
										use_bias=True,
										trainable=trainable, name='q_dense_h3_agent'+str(nn))
				q_value = tf.layers.dense(inputs=q_hidden_3, units=1,
										# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
										# bias_initializer=tf.constant_initializer(0.00),  # biases
										use_bias=True,
										trainable=trainable, name='dense_h4_agent'+str(nn))
				q_values_output[nn] = q_value
		
			q_value_list = q_values_output.values()
			q_values = tf.concat([tensor for tensor in q_value_list], axis=1)

			#tge hypernetworks take the global state s as the input and outputs the weights of the feedforward network
			s = tf.reshape(self.global_state, [-1, self.len_state_single*self.num_of_agents])
			w1 = tf.layers.dense(s, self.num_of_agents * h3,
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_w1')
			w2 = tf.layers.dense(s, h3 * 1, 
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_w2')
			b1 = tf.layers.dense(s, h3,
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_b1')
			b2_h = tf.layers.dense(s, h3,  activation=tf.nn.relu,
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_b2_h')
			b2 = tf.layers.dense(b2_h, 1, 
									# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
									# bias_initializer=tf.constant_initializer(0.00),  # biases
									use_bias=True,
									trainable=trainable, name='dense_b2')
			w1 = tf.abs(w1)
			w1 = tf.reshape(w1, [-1, self.num_of_agents, h3])
			w2 = tf.abs(w2)
			w2 = tf.reshape(w2, [-1, h3, 1])
			q_values = tf.reshape(q_values, [-1,1,self.num_of_agents])
			q_hidden = tf.nn.elu(tf.reshape(tf.matmul(q_values, w1),[-1,h3]) )  + b1
			q_hidden = tf.reshape(q_hidden, [-1,1,h3])
			q_value_mix = tf.reshape(tf.matmul(q_hidden, w2),[-1,1]) + b2

			return q_values_output, q_value_mix

	def get_q_values(self, sess, global_state, obs, dis_actions, con_actions):
		return sess.run(self.output_var, feed_dict={self.global_state : global_state, self.obs: obs, self.dis_actions : dis_actions, self.con_actions : con_actions})[0]

	def get_q_values_target(self, sess, global_state, obs, dis_actions, con_actions):
		return sess.run(self.output_var_target, feed_dict={self.global_state : global_state, self.obs: obs, self.dis_actions : dis_actions, self.con_actions : con_actions})[0]

	def get_q_value_mix_target(self, sess, global_state, obs, dis_actions, con_actions):
		return sess.run(self.output_var_target, feed_dict={self.global_state : global_state, self.obs: obs, self.dis_actions : dis_actions, self.con_actions : con_actions})[1]

	def get_gradients(self, sess, global_state, obs, dis_actions, con_actions):
		return sess.run(self.action_gradients, feed_dict={self.global_state : global_state, self.obs: obs, self.dis_actions : dis_actions, self.con_actions : con_actions})

def InitialiseActor(g_1, T_max, num_of_agents, num_of_zones):
	with g_1.as_default():
		global_s = tf.placeholder(shape=[None, num_of_agents, num_of_zones*2+T_max+1], dtype=tf.float32)
		obs = tf.placeholder(shape=[None, num_of_agents, num_of_zones], dtype=tf.float32)
		dis_actions = tf.placeholder(shape=[None, num_of_agents, 4], dtype=tf.float32)
		ConNN = Actor(T_max=T_max, num_of_agents=num_of_agents, num_of_zones=num_of_zones, s=global_s, obs=obs, dis_actions=dis_actions)
		ConNN.setCompGraph()
	return ConNN

def InitialiseCritic(g_1, T_max, num_of_agents, num_of_zones):
	with g_1.as_default():
		global_s = tf.placeholder(shape=[None, num_of_agents, num_of_zones*2+T_max+1], dtype=tf.float32)
		obs = tf.placeholder(shape=[None, num_of_agents, num_of_zones], dtype=tf.float32)
		con_actions = tf.placeholder(shape=[None, num_of_agents, 1], dtype=tf.float32)
		dis_actions = tf.placeholder(shape=[None, num_of_agents, 4], dtype=tf.float32)
		DisNN = Critic(T_max=T_max, num_of_agents=num_of_agents, num_of_zones=num_of_zones, s=global_s, obs=obs, dis_actions=dis_actions, con_actions=con_actions)
		DisNN.setCompGraph()
	return DisNN