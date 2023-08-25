# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:32:08 2019

@author: jjling.2018
"""

import json
from itertools import groupby
import tensorflow as tf
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
import sys
import random
import time
from collections import defaultdict
from collections import deque
from optparse import OptionParser
from pathos.multiprocessing import ThreadingPool as Pool
import os 
from utilis import *
from policyNet import HA_Actor
from criticNet import HA_Critic

																						   
def episodeLoop(env, eps_threshold, options, all_static_info, sess):
	edges = all_static_info["edges"]
	polNNs = all_static_info["polNNs"]
	length = all_static_info["length"]
	T_min = all_static_info["T_min"]
	T_max = all_static_info["T_max"]
	num_of_agents = all_static_info["num_of_agents"]
	num_of_zones = all_static_info["num_of_zones"]
	GOALS = all_static_info["GOALS"]
	uniqueGOALS = all_static_info["uniqueGOALS"]
	num_of_GOALS = all_static_info["num_of_GOALS"]
	GOALS_index = all_static_info["GOALS_index"]
	cap = all_static_info["cap"] 
	landmarks = all_static_info["landmarks"]
 
	obsdict = {} 
	actdict = {}
	timeTodestdict = {}
	retdict = {}
	nextretdict = {}

	# Each trajectory will have at most 100 time steps
	T = options.T

	# Set the discount factor for the problem
	discount = options.discount

	#new episode
	flags = [False] * num_of_agents
	flags_landmark = [[False] * 5 for _ in range(num_of_agents)] 
	#observations
	states = [[] for _ in range(num_of_agents)]

	states_com2 = [[] for _ in range(num_of_agents)]
	states_com3 = [[] for _ in range(num_of_agents)]
	states_com4 = [[] for _ in range(num_of_agents)]
	observations = []
	global_states = []
	# store sampled actions i
	actions = []
	actionZones = [[] for _ in range(num_of_agents)]
	actionTimes = [[] for _ in range(num_of_agents)]
	nusParas = []
	entropys = [[] for _ in range(num_of_agents)]

	# Empirical return for each sampled state-action pair
	rewards = [[] for _ in range(num_of_agents)]
	num_collision = 0

	env.reset()
	
	for steps in range(0, T):   
		action_matrix = [] 
		actionIDs = [] 
		actionZone = []
		zoneIDs = []
		actionTime = []
		obs = []
		goals = []
		destIDs = []
		timeDests = []
		nusPara = []
		entropy = []
		global_state = []
		hybrid_actions = {}
		
		for nn in range(0, num_of_agents): 
			obs_single = env.vector_observations[nn]
			agentID = int(obs_single[0])
			zoneID = int(obs_single[1])
			destID = int(obs_single[2])
			timeDest = int(obs_single[3])
			goal = int(obs_single[4])
			firstVisit = 1 if goal != -1 else 0
			zoneIDs.append(zoneID)
			goals.append(goal)
			destIDs.append(destID)
			timeDests.append(timeDest)
	
			possibleneighbourlist = []
			for element in range(5, len(obs_single), 2):
				if obs_single[element] != -1:
					possibleneighbourlist.extend(obs_single[element:element+2])                                                                                                     
			obsOnehot = TransObsOnehot(possibleneighbourlist, num_of_zones)
			obs.append(obsOnehot)
			global_state.append(to_one_hot(zoneID, num_of_zones)+to_one_hot(destID, num_of_zones)+to_one_hot(timeDest, T_max)+[firstVisit])
			
			hybrid_actions[nn] = [[],[]]


		obs_input = np.asarray(obs).reshape((1, num_of_agents, num_of_zones))
		global_state_input = np.asarray(global_state).reshape((1, num_of_agents, num_of_zones*2+T_max+1))
		
		for all_act in range(4):	
			dis_actions_input = np.asarray([to_one_hot(all_act, 4)] * num_of_agents).reshape((1, num_of_agents, 4))						
			
			con_actions = ConNN.getAction(sess, global_state_input, obs_input, dis_actions_input)        
			
			con_actions_input = np.asarray([xx[0][0] for xx in con_actions.values()]).reshape((1, num_of_agents, 1))	
			# con_actions_input = np.asarray([0 for xx in con_actions.values()]).reshape((1, num_of_agents, 1))	
			q_values = DisNN.get_q_values(sess, global_state_input, obs_input, dis_actions_input, con_actions_input)

			for nn in range(0, num_of_agents): 
				hybrid_actions[nn][0].append(con_actions[nn][0][0])
				hybrid_actions[nn][1].append(q_values[nn][0][0])


		for nn in range(0, num_of_agents): 	
			if(destIDs[nn] == -1 and timeDests[nn] == -1 and (goals[nn] != -1 and goals[nn] != zoneIDs[nn])):   			
				possible_actions = [p_a for p_a in range(len(edges[zoneIDs[nn]]))]
				if np.random.rand() < eps_threshold:
					a = np.random.choice(possible_actions)
				#max q_value
				else:
					a = possible_actions[np.argmax([hybrid_actions[nn][1][p_a] for p_a in possible_actions])]
				   
				nu = hybrid_actions[nn][0][a] + 0.1*np.random.randn(1)[0]
				nu = np.clip(nu, 0, 1)
				nusPara.append(nu)
				m = T_max - T_min
				timeTodest = T_min + np.random.binomial(m, nu)
				actionIDs.append(to_one_hot(a, 4))
				actionZone.append(edges[zoneIDs[nn]][a])
				actionTime.append(timeTodest-T_min)
				action_matrix.append([edges[zoneIDs[nn]][a], timeTodest])
			else:
				actionIDs.append(to_one_hot(-1, 4))
				actionZone.append(-1)
				actionTime.append(-1)
				action_matrix.append([-1.0, -1.0])
				nusPara.append(-1)


		action_matrix = np.array(action_matrix)      
		env.step(action_matrix)

		# #collect states and actions
		observations.append(obs)  
		global_states.append(global_state)
		nusParas.append(nusPara)
		actions.append(actionIDs)
		
		for nn in range(num_of_agents):
			states[nn].append(zoneIDs[nn])
			actionZones[nn].append(actionZone[nn])

		# #collect rewards
		# #penalty for violating collison
		penaltyAgents = {}
		for group in groupby(sorted(enumerate(zoneIDs), key=lambda x: x[1]), lambda x: x[1]):
			zone = group[0]
			zone_agents = [x[0] for x in group[1]]

			if len(zone_agents) > cap[zone]:
				num_collision += 1
				penalty = -options.penalty
			else:
				penalty = 0
		   
			for agent in zone_agents:
				penaltyAgents[agent] = penalty

		for nn in range(0, num_of_agents): 
			if zoneIDs[nn] == GOALS[nn] and not flags[nn]:
				stepReward = options.finalReward
				flags[nn] = True
			elif zoneIDs[nn] == GOALS[nn] and flags[nn]:
				stepReward = 0
			else:
				if zoneIDs[nn] in landmarks:      
					lanmard_id = landmarks.index(zoneIDs[nn])            
					if not flags_landmark[nn][lanmard_id] :
						stepReward = options.landmarkReward
						flags_landmark[nn][lanmard_id] = True
					else:
						stepReward = -options.timeReward
				else:
					stepReward = -options.timeReward	

			if flags[nn] == False:	   
				next_state_single = env.vector_observations[nn][1:5]
				phi_next_state = potential(length, int(next_state_single[0]), int(next_state_single[1]), int(next_state_single[2]), int(next_state_single[3]))  
				phi_state = potential(length, zoneIDs[nn], destIDs[nn], timeDests[nn], goals[nn])       
				shaping = discount * phi_next_state - phi_state                                 
			else:
				shaping = 0                       			
			rewards[nn].append(stepReward + penaltyAgents[nn])
			
		if all(flags):                     
			break  
	
	total_len = 0
	for nn in range(num_of_agents):
		path = states[nn]
		goal_agent = GOALS[nn]
		if path[-1] == goal_agent:
			path = [z for x, y in groupby(path) for z in y if z!=goal_agent] + [goal_agent]
		else:
			path = [z for x, y in groupby(path) for z in y if z!=goal_agent] 
		total_len += len(path) -1               
	
	resultSavePath = './ha_log/results/'+str(options.mapSize)+'/agents'+str(options.numAgents)+'/'+options.experimentname
	with open(resultSavePath, 'a+') as f:
		f.write(json.dumps(states))
		f.write('\n')

	sample_len = len(rewards[0])
	global_rewards = np.asarray(rewards).reshape((num_of_agents,sample_len))
	global_rewards = list(np.sum(global_rewards,axis=0)[1:])
	#episode ends, samples for training critic network
	samples_global_states = np.asarray(global_states).reshape((sample_len, num_of_agents, num_of_zones*2+T_max+1))
	samples_observations = np.asarray(observations).reshape((sample_len, num_of_agents, num_of_zones))
	samples_dis_actions = np.asarray(actions).reshape((sample_len, num_of_agents, 4))
	samples_con_actions = np.asarray(nusParas).reshape((sample_len, num_of_agents, 1))
	samples_states = np.asarray(states).reshape((num_of_agents,sample_len)).T
	
	return (all(flags), total_len, num_collision, samples_states, samples_global_states, samples_observations, samples_dis_actions, samples_con_actions, global_rewards)
																			   
def trainModel(options, cap, edges, g_1, ConNN, DisNN, length, listofenvironments, T_min, T_max, num_of_agents, num_of_zones, GOALS, landmarks): 
	uniqueGOALS = list(np.unique(GOALS))
	num_of_GOALS = len(uniqueGOALS)
	GOALS_index = {yy : xx for xx, yy in enumerate(uniqueGOALS)}
	with tf.Session(graph=g_1, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('./ha_log/tflogs/'+str(options.mapSize)+'/agents'+str(options.numAgents)+'/'+options.experimentname, sess.graph)
		N = options.N
		T = options.T
		n_itr = options.n_itr
		discount = options.discount
		paths = []
		Collision = []                                             
		pool = Pool(processes=len(listofenvironments)+4)

		all_static_info = {}
		all_static_info["edges"] = edges
		all_static_info["polNNs"] = polNNs
		all_static_info["length"] = length
		all_static_info["T_min"] = T_min
		all_static_info["T_max"] = T_max
		all_static_info["num_of_agents"] = num_of_agents
		all_static_info["num_of_zones"] = num_of_zones
		all_static_info["GOALS"] = GOALS
		all_static_info["uniqueGOALS"] = uniqueGOALS
		all_static_info["num_of_GOALS"] = num_of_GOALS
		all_static_info["GOALS_index"] = GOALS_index
		all_static_info["cap"] = cap
		all_static_info["landmarks"] = landmarks

		for i in range(0,n_itr): 
			QMIXParameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qmix/eval')
			QMIXTargetParameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qmix/target') 
			MuParameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Mu/eval')
			MuTargetParameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Mu/target') 
			soft_replacement = [tf.assign(t, 0.99 * t + 0.01* e) for t, e in zip(QMIXTargetParameters,QMIXParameters)]
			sess.run(soft_replacement)
			soft_replacement = [tf.assign(t, 0.99 * t + 0.01* e) for t, e in zip(MuTargetParameters,MuParameters)]
			sess.run(soft_replacement) 

			#new iteration
			print("Iteration->",i)
			resultSavePath = './ha_log/results/'+str(options.mapSize)+'/agents'+str(options.numAgents)
			if not os.path.exists(resultSavePath):
				os.makedirs(resultSavePath)
			with open(resultSavePath+'/'+options.experimentname, 'a+') as f:
				f.write('Iteration'+str(i)+'\n')
			
			#did not consider the case that last state is terminal state.
			all_samples_global_states = []
			all_samples_next_global_states = []
			all_samples_last_global_states = []
			all_samples_observations = []
			all_samples_next_observations = []
			all_samples_last_observations = []
			all_samples_dis_actions = []
			all_samples_next_dis_actions = []
			all_samples_last_dis_actions = []
			all_samples_con_actions = []
			all_samples_next_con_actions = []
			all_samples_last_con_actions = []
			all_rets = []
			all_last_rets = []
			all_samples_next_states = [] 

			eps_threshold = 0.05 + (0.9 - 0.05) * np.exp(-1. * i / 300)

			result = pool.amap(episodeLoop, listofenvironments, [eps_threshold]*len(listofenvironments), [options]*len(listofenvironments), [all_static_info]*len(listofenvironments), [sess]*len(listofenvironments))
			all_dictionaries_all_episodes = result.get()

			for data_from_env in range(0,N):  

				all_reach, total_len_eps, num_collision_eps, samples_states, samples_global_states, samples_observations, samples_dis_actions, samples_con_actions, rets = all_dictionaries_all_episodes[data_from_env]	
				
				all_samples_global_states.append(samples_global_states[0:-2,:,:])
				all_samples_next_global_states.append(samples_global_states[1:-1,:,:])
				all_samples_observations.append(samples_observations[0:-2,:,:])
				all_samples_next_observations.append(samples_observations[1:-1,:,:])
				all_samples_dis_actions.append(samples_dis_actions[0:-2,:,:])
				all_samples_next_dis_actions.append(samples_dis_actions[1:-1,:,:])
				all_samples_con_actions.append(samples_con_actions[0:-2,:,:])
				all_samples_next_con_actions.append(samples_con_actions[1:-1,:,:])
				all_rets.append(rets[0:-1])
				all_samples_next_states.append(samples_states[1:-1,:])

				if all_reach:
			
					all_samples_last_global_states.append(samples_global_states[-2:-1,:,:])
					all_samples_last_observations.append(samples_observations[-2:-1,:,:])
					all_samples_last_dis_actions.append(samples_dis_actions[-2:-1,:,:])			 
					all_samples_last_con_actions.append(samples_con_actions[-2:-1,:,:])
					all_last_rets.append([rets[-1]])

				paths.append(total_len_eps)
				Collision.append(num_collision_eps)  
			
			all_samples_global_states = np.concatenate(all_samples_global_states, axis=0)
			all_samples_next_global_states = np.concatenate(all_samples_next_global_states, axis=0)
			
			all_samples_observations = np.concatenate(all_samples_observations, axis=0)
			all_samples_next_observations = np.concatenate(all_samples_next_observations, axis=0)
			
			all_samples_dis_actions = np.concatenate(all_samples_dis_actions, axis=0)
			all_samples_next_dis_actions = np.concatenate(all_samples_next_dis_actions, axis=0)
					
			all_samples_con_actions = np.concatenate(all_samples_con_actions, axis=0)
			all_samples_next_con_actions = np.concatenate(all_samples_next_con_actions, axis=0)
			
			all_rets = np.concatenate(all_rets, axis=0)
			  
			all_samples_next_states = np.concatenate(all_samples_next_states, axis=0)

			if len(all_last_rets) != 0:
				all_samples_last_global_states = np.concatenate(all_samples_last_global_states, axis=0)
				all_samples_last_observations = np.concatenate(all_samples_last_observations, axis=0)
				all_samples_last_dis_actions = np.concatenate(all_samples_last_dis_actions, axis=0)
				all_samples_last_con_actions = np.concatenate(all_samples_last_con_actions, axis=0)
				all_last_rets = np.concatenate(all_last_rets, axis=0)

			hybrid_actions = {}
			for nn in range(0, num_of_agents):  
				hybrid_actions[nn] = [[],[]]
			sample_len = all_samples_observations.shape[0]
			for all_act in range(4):	
				dis_actions_input = np.asarray([to_one_hot(all_act, 4)] * num_of_agents * sample_len).reshape((sample_len, num_of_agents, 4))										
				con_actions = ConNN.getAction_target(sess, all_samples_next_global_states, all_samples_next_observations, dis_actions_input)        				
				con_actions_input = np.asarray([xx[:,0] for xx in con_actions.values()]).T.reshape((sample_len, num_of_agents, 1))
				q_values_target = DisNN.get_q_values_target(sess, all_samples_next_global_states, all_samples_next_observations, dis_actions_input, con_actions_input)

				for nn in range(0, num_of_agents): 
					hybrid_actions[nn][0].append(con_actions[nn][:,0])
					hybrid_actions[nn][1].append(q_values_target[nn][:,0])

			for nn in range(0, num_of_agents): 
				hybrid_actions[nn][0] = np.asarray(hybrid_actions[nn][0]).T
				hybrid_actions[nn][1] = np.asarray(hybrid_actions[nn][1]).T

			dis_actions_input = []
			con_actions_input = []

			for tt in range(sample_len):
				tmp_dic_actions = []
				tmp_con_actions = []
				for nn in range(0, num_of_agents):
					if all_samples_next_con_actions[tt][nn][0] == -1:
						a = -1 
						nu = -1                       
					else:
						a = np.argmax(hybrid_actions[nn][1][tt][0:len(edges[all_samples_next_states[tt][nn]])])	
						nu = hybrid_actions[nn][0][tt][a]
					tmp_dic_actions.append(to_one_hot(a, 4))
					tmp_con_actions.append(nu)
				dis_actions_input.append(tmp_dic_actions)
				con_actions_input.append(tmp_con_actions)

			dis_actions_input = np.asarray(dis_actions_input).reshape((sample_len, num_of_agents, 4))
			con_actions_input = np.asarray(con_actions_input).reshape((sample_len, num_of_agents, 1))
			# con_actions_input = np.asarray([0]*sample_len*num_of_agents).reshape((sample_len, num_of_agents, 1))	

			q_value_mix_next = DisNN.get_q_value_mix_target(sess, all_samples_next_global_states, all_samples_next_observations, dis_actions_input, con_actions_input)
			# q_value_mix_next = DisNN.get_q_value_mix_target(sess, all_samples_next_global_states, all_samples_next_observations, all_samples_next_dis_actions, all_samples_next_con_actions)
			#starting training critic
			inpdict = {}
			if len(all_last_rets) != 0:
				inpdict[DisNN.global_state] = np.concatenate([all_samples_global_states, all_samples_last_global_states], axis=0)				
				inpdict[DisNN.obs] = np.concatenate([all_samples_observations, all_samples_last_observations], axis=0)	
				inpdict[DisNN.dis_actions] = np.concatenate([all_samples_dis_actions, all_samples_last_dis_actions], axis=0)						
				inpdict[DisNN.con_actions] = np.concatenate([all_samples_con_actions, all_samples_last_con_actions], axis=0)	
				inpdict[DisNN.r] = np.concatenate([all_rets.reshape((len(all_rets),1)), all_last_rets.reshape((len(all_last_rets),1))], axis=0) 	
				inpdict[DisNN.q_value_mix_next] =  np.concatenate([q_value_mix_next, np.asarray([0]*len(all_last_rets)).reshape((len(all_last_rets),1))], axis=0) 
			
			else:
				inpdict[DisNN.global_state] = all_samples_global_states				
				inpdict[DisNN.obs] = all_samples_observations	
				inpdict[DisNN.dis_actions] = all_samples_dis_actions						
				inpdict[DisNN.con_actions] = all_samples_con_actions	
				inpdict[DisNN.r] = all_rets.reshape((len(all_rets),1))	
				inpdict[DisNN.q_value_mix_next] =  q_value_mix_next
			
			sess.run(DisNN.learning_step, feed_dict=inpdict)
			loss1_after = sess.run(DisNN.loss, feed_dict=inpdict)
			
			#starting training actor
			gradients = DisNN.get_gradients(sess, all_samples_global_states, all_samples_observations, all_samples_dis_actions, all_samples_con_actions)[0]

			inpdicts = {}
			all_samples_global_states_single = {}
			all_samples_observations_single = {}
			all_samples_dis_actions_single = {}
			gradients_single ={}

			for nn in range(num_of_agents):
				inpdicts[nn] = {}
				all_samples_global_states_single[nn] = []
				all_samples_observations_single[nn] = []
				all_samples_dis_actions_single[nn] = []
				gradients_single[nn] =[]

			sample_len = all_samples_con_actions.shape[0]

			for tt in range(sample_len):
				for nn in range(num_of_agents):
					if all_samples_con_actions[tt][nn][0] != -1:
						all_samples_global_states_single[nn].append(all_samples_global_states[tt,:,:])
						all_samples_observations_single[nn].append(all_samples_observations[tt,:,:])
						all_samples_dis_actions_single[nn].append(all_samples_dis_actions[tt,:,:])
						gradients_single[nn].append(gradients[tt,:,:])
	
			for nn in range(num_of_agents):
				inpdicts[nn][ConNN.global_state] = np.asarray(all_samples_global_states_single[nn]).reshape((-1, num_of_agents, num_of_zones*2+T_max+1))			
				inpdicts[nn][ConNN.obs] = np.asarray(all_samples_observations_single[nn]).reshape((-1, num_of_agents, num_of_zones))
				inpdicts[nn][ConNN.dis_actions] = np.asarray(all_samples_dis_actions_single[nn]).reshape((-1, num_of_agents, 4))					
				inpdicts[nn][ConNN.action_gradients] = np.asarray(gradients_single[nn]).reshape((-1, num_of_agents, 1))
				sess.run(ConNN.learning_step[nn], feed_dict=inpdicts[nn])

				
			summary = tf.Summary()            
			summary.value.add(tag='summaries/length_of_path', simple_value = np.mean(paths[i*N : (i+1)*N]))
			summary.value.add(tag='summaries/num_collision', simple_value = np.mean(Collision[i*N : (i+1)*N]))
			summary.value.add(tag='summaries/actor_training_loss', simple_value = (loss2_before+loss2_after)/2)
			summary.value.add(tag='summaries/critic_training_loss', simple_value = loss1_after)		  
			writer.add_summary(summary, i)          
			writer.flush()     
		saver = tf.train.Saver()
		save_path = './ha_log/trainedModel/'+str(options.mapSize)+'/agents'+str(options.numAgents)+'/'+options.experimentname
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		saver.save(sess, save_path+'/model.ckpt') 
		for env in listofenvironments:				
			env.close()
	
def main():
	parser = OptionParser()
	#training arguments
	parser.add_option("-t", "--time", type='int', dest='T', default=500)
	parser.add_option("-b", "--batch", type='int', dest='N', default=8)
	parser.add_option("-i", "--iteration", type='int', dest='n_itr', default=500)    
	parser.add_option("-d", "--discount", type='float', dest='discount', default=0.99)   
	parser.add_option("-l", "--learningrate", type='float', dest='lr', default=0.001)
	#instance arguments
	parser.add_option("-z", "--size", type='str', dest='mapSize', default='5x5')
	parser.add_option("-m", "--map", type='str', dest='mapType', default='landmark', help='either open or landmark')   
	parser.add_option("-n", "--numAgents", type='int', dest='numAgents', default=6)
	parser.add_option("-k", "--numLandmarks", type='int', dest='numLandmarks', default=0)
	parser.add_option("-s", "--instanceID", type='int', dest='instanceID', default=0) 	
	parser.add_option("-x", "--experimentRun", type='int', dest='experimentRun', default=0)   		
	#reward arguments
	parser.add_option("-r", "--timeReward", type='float', dest='timeReward', default=1)
	parser.add_option("-q", "--landmarkReward", type='float', dest='landmarkReward', default=25)
	# parser.add_option("-o", "--noopReward", type='float', dest='noopReward', default=1)
	parser.add_option("-f", "--finalReward", type='float', dest='finalReward', default=100)
	parser.add_option("-p", "--penalty", type='float', dest='penalty', default=50)
	(options, args) = parser.parse_args()  

	options.modelPath = '../Models/'+str(options.mapSize)+'.model'
	options.experimentname = str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_agents'+str(options.numAgents)+'_ex'+str(options.instanceID)+'_run'+str(options.experimentRun)+'.txt'
	options.configPath = '../Configs/'+str(options.mapSize)+'/agents'+str(options.numAgents)+'/'+str(options.mapSize)+'_ex'+str(options.instanceID)+'.txt'
	options.landmarkConfigPath = '../data/landmarkFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'.txt'

	T_min, T_max, max_cap, num_of_zones, num_of_agents, start_zones, goal_zones, STARTS, GOALS, start_goal, landmarks = readConfiguration(options.configPath, options.landmarkConfigPath)
	cap = capInitialise(num_of_agents, num_of_zones, max_cap, start_goal, randomSeed=666)
	length = shortestPath(options.modelPath)
	edges = constructGraph(options.modelPath)
	listofenvironments = envInitialise(options, num_of_agents, STARTS, GOALS, edges)

	g_1 = tf.Graph()   
	ConNN = InitialiseActor(g_1, T_max, num_of_agents, num_of_zones)
	DisNN = InitialiseCritic(g_1, T_max, num_of_agents, num_of_zones)

	trainModel(options, cap, edges, g_1, ConNN, DisNN, length, listofenvironments, T_min, T_max, num_of_agents, num_of_zones, GOALS, landmarks)

if __name__ == '__main__':
	main()