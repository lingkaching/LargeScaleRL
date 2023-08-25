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

class Actor:
	def __init__(self, bi_n, learningRate, num_of_GOALS, input_var, input_var1, numUnitsPerHLayer, neighbours, zoneID = None):
		self.num_of_GOALS = num_of_GOALS
		self.bi_n = bi_n
		self.zoneID = zoneID
		self.input_var = input_var
		self.input_var1 = input_var1
		self.numUnitsPerHLayer = numUnitsPerHLayer
		self.neighbours = neighbours    
		with tf.variable_scope('Actor'):   
			self.output_var = self.initPolNN(scope='eval', trainable=True)		
		#action
		self.act_var = None
		#reward
		self.ret_var = None
		self.nextret_var = None 
		self.phi = None
		self.selected_dis_action = None
		self.selected_con_action = None
		self.returnAndSelectedPolicy = None
		self.zoneBasedVal = None
		self.finalObj = None
		self.val1 = None
		self.val2 = None
		self.val3 = None
		self.learning_rate = learningRate
		self.optimizer = None
		self.learning_step = None
		self.gamma = 0.99
		
	def setCompGraph(self):
		with tf.variable_scope('ActorTraning'):
			self.act_var = tf.placeholder(shape=[None, self.num_of_GOALS, self.neighbours], dtype=tf.float32)
			self.ret_var = tf.placeholder(shape=[1,None], dtype=tf.float32)
			self.nextret_var = tf.placeholder(shape=[1,None], dtype=tf.float32)
			self.phi = tf.placeholder(shape=[1,None], dtype=tf.float32)		
			temp1 = 0
			temp2 = 0
			for nn in range(self.num_of_GOALS):
				temp1 = tf.add(self.output_var[nn] * self.act_var[:,nn], temp1)			
				temp2 = tf.add(self.output_var[nn+self.num_of_GOALS] * tf.reduce_sum(self.act_var[:,nn],axis=1,keepdims=True), temp2)			
			self.selected_dis_action = tf.reduce_sum(temp1, axis=1)                      			
			self.selected_con_action = tf.reduce_sum(temp2, axis=1)						
			self.val1 = tf.log(self.selected_dis_action + 1e-8) * self.ret_var
			self.val2 = tf.add(tf.log(self.selected_con_action + 1e-8 ),tf.negative(tf.log1p(1e-8 + tf.negative(self.selected_con_action))))* (self.nextret_var * self.phi)	 
			self.val3 = tf.log1p(1e-8 + tf.negative(self.selected_con_action))  * (self.nextret_var *  self.bi_n)			
			self.zoneBasedVal = tf.add(self.val1, self.gamma*(tf.add(self.val2, self.val3)))			
			self.finalObj = tf.reduce_sum(self.zoneBasedVal, axis=1) 
			self.loss = -1*(self.finalObj)

			# Defining Optimizer
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

			# Update Gradients
			self.learning_step = (self.optimizer.minimize(self.loss))

	def initPolNN(self, scope, trainable):
		OUTPUT = {}
		with tf.variable_scope(scope):       
			l_in = tf.layers.dense(inputs=self.input_var, units=self.numUnitsPerHLayer, activation=tf.nn.relu, trainable=trainable, name=str(self.zoneID)+"-DenseReLu_1")           			
			l_in_norm = tf.contrib.layers.layer_norm(inputs=l_in, trainable=trainable, scope=str(self.zoneID)+'norm1')           	
			l_hid_1 = tf.layers.dense(inputs=l_in_norm, units=self.numUnitsPerHLayer, activation=tf.nn.relu, trainable=trainable, name=str(self.zoneID)+"-DenseReLu_2")           			
			l_hid_1_norm = tf.contrib.layers.layer_norm(inputs=l_hid_1, trainable=trainable, scope=str(self.zoneID)+'norm2')           			
			l_hid_2 = tf.layers.dense(inputs=l_hid_1_norm, units=self.numUnitsPerHLayer, activation=tf.nn.relu, trainable=trainable, name=str(self.zoneID)+"-DenseReLu_3")                   			
			l_hid_2_norm = tf.contrib.layers.layer_norm(inputs=l_hid_2, trainable=trainable, scope=str(self.zoneID)+'norm3')
						
			for nn in range(self.num_of_GOALS):			
				l_discrete_out = tf.layers.dense(inputs=l_hid_2_norm, units=self.neighbours, activation=tf.nn.softmax, trainable=trainable, name=str(self.zoneID)+"-Softmax"+str(nn))        			
				l_continuous_out = tf.layers.dense(inputs=tf.concat([self.input_var1[:,nn], l_hid_2_norm], 1), units=1, activation=tf.nn.sigmoid, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(), trainable=trainable, name=str(self.zoneID)+"-Sigmoid"+str(nn))               			
				OUTPUT[nn] = l_discrete_out			
				OUTPUT[nn+self.num_of_GOALS] = l_continuous_out			
			return OUTPUT

	def getAction(self, sess, GoalObs, DisActions):
		return sess.run(self.output_var, feed_dict={self.input_var : GoalObs, self.input_var1 : DisActions})

h1 = 64
h2 = 64
h3 = 64
h4 = 64
h5 = 64

class HA_Actor:
	def __init__(self, T_max, num_of_agents, num_of_zones, s, obs, dis_actions):
		self.num_of_agents = num_of_agents
		self.num_of_zones = num_of_zones
		self.len_state_single = num_of_zones*2+T_max+1
		self.obs = obs
		self.global_state = s
		self.dis_actions = dis_actions
			  
		self.learning_rate = None
		self.optimizer = None
		self.learning_step_single = None
		self.learning_step = {}

		with tf.variable_scope('Mu'):
			self.output_var = self.generate_mu(scope='eval', trainable=True)
			self.output_var_target = self.generate_mu(scope='target', trainable=False)

		self.model_weights = {}
		for nn in range(self.num_of_agents): 
			self.model_weights[nn] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Mu/eval/agent'+str(nn))
		
	def setCompGraph(self):
		with tf.variable_scope('optimisation'):
			self.action_gradients = tf.placeholder(shape=[None, self.num_of_agents, 1],dtype=tf.float32)
			for nn in range(self.num_of_agents):
				action_gradients_single = tf.slice(self.action_gradients, [0, nn, 0], [-1, 1, 1])
				
				action_gradients_single = tf.reshape(tensor=action_gradients_single, shape=(-1, 1))
				
				self.parameter_gradietns_single = tf.gradients(self.output_var[nn], self.model_weights[nn], -action_gradients_single)

				self.gradients_single = zip(self.parameter_gradietns_single, self.model_weights[nn])

				# Learning Rate
				self.learning_rate = 0.001

				# Defining Optimizer
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

				# Update Gradients
				self.learning_step_single = self.optimizer.apply_gradients(self.gradients_single)

				self.learning_step[nn] = self.learning_step_single

	def generate_mu(self, scope, trainable):
		with tf.variable_scope(scope):
			con_action_output = {}
			for nn in range(self.num_of_agents):    

				with tf.variable_scope('agent'+str(nn)):

					state_single = tf.slice(self.global_state, [0, nn, 0], [-1, 1, self.len_state_single])
					state_single = tf.reshape(tensor=state_single, shape=(-1, self.len_state_single))

					obs_single = tf.slice(self.obs, [0, nn, 0], [-1, 1, self.num_of_zones])
					obs_single = tf.reshape(tensor=obs_single, shape=(-1, self.num_of_zones))
					
					dis_action_single = tf.slice(self.dis_actions, [0, nn, 0], [-1, 1, 4])
					dis_action_single = tf.reshape(tensor=dis_action_single, shape=(-1, 4))
	
	
					# generate mu network
					mu_hidden_1 = tf.layers.dense(inputs=tf.concat([state_single, obs_single, dis_action_single], 1), units=h1, activation=tf.nn.relu,
											# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
											# bias_initializer=tf.constant_initializer(0.00),  # biases
											use_bias=True,
											trainable=trainable, name='mu_dense_h1_agent'+str(nn))
					mu_hidden_2 = tf.layers.dense(inputs=mu_hidden_1, units=h2, 
											# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
											# bias_initializer=tf.constant_initializer(0.00),  # biases
											use_bias=True,
											trainable=trainable, name='mu_dense_h2_agent'+str(nn))
					con_action_single = tf.layers.dense(inputs=mu_hidden_2, units=1, activation=tf.nn.sigmoid,
											# kernel_initializer=tf.random_normal_initializer(0., .01),    # weights
											# bias_initializer=tf.constant_initializer(0.00),  # biases
											use_bias=True,
											trainable=trainable, name='mu_dense_h3_agent'+str(nn))
	
					con_action_output[nn] = con_action_single
			
			return con_action_output

	def getAction(self, sess, global_state, obs, dis_actions):
		return sess.run(self.output_var, feed_dict={self.global_state : global_state, self.obs : obs, self.dis_actions : dis_actions})

	def getAction_target(self, sess, global_state, obs, dis_actions):
		return sess.run(self.output_var_target, feed_dict={self.global_state : global_state, self.obs : obs, self.dis_actions : dis_actions})