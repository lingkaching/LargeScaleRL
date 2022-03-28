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
	def __init__(self, bi_n, learningRate, num_of_GOALS, input_var, input_var1, input_var2, numUnitsPerHLayer, neighbours, zoneID = None):
		self.num_of_GOALS = num_of_GOALS
		self.bi_n = bi_n
		self.zoneID = zoneID
		self.input_var = input_var
		self.input_var1 = input_var1
		self.input_var2 = input_var2
		self.numUnitsPerHLayer = numUnitsPerHLayer
		self.neighbours = neighbours    
		with tf.variable_scope('Actor'):   
			self.output_var = self.initPolNN(scope='eval', trainable=True)
		self.act_var = None
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
		self.gvs = None
		self.capped_gvs = None
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
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
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
				l_discrete_out_1 = tf.layers.dense(inputs=l_hid_2_norm, units=self.neighbours, trainable=trainable, name=str(self.zoneID)+"-Softmax"+str(nn))  
				l_discrete_out = tf.exp(l_discrete_out_1-tf.reduce_max(l_discrete_out_1))*self.input_var2[:,nn]/tf.reduce_sum(tf.exp(l_discrete_out_1-tf.reduce_max(l_discrete_out_1))*self.input_var2[:,nn])      
				l_continuous_out = tf.layers.dense(inputs=tf.concat([self.input_var1[:,nn], l_hid_2_norm], 1), units=1, activation=tf.nn.sigmoid, kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.zeros_initializer(), trainable=trainable, name=str(self.zoneID)+"-Sigmoid"+str(nn))               
				OUTPUT[nn] = l_discrete_out			
				OUTPUT[nn+self.num_of_GOALS] = l_continuous_out			
			return OUTPUT
	def getAction(self, sess, GoalObs, DisActions, Valid_DisActions):
		return sess.run(self.output_var, feed_dict={self.input_var : GoalObs, self.input_var1 : DisActions, self.input_var2 : Valid_DisActions})

		   
