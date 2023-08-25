# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:23:27 2019

@author: jjling.2018
"""
# from __future__ import print_function
import json
from itertools import groupby
import tensorflow as tf
import networkx as nx 
import numpy as np
import sys
import random
import time
from collections import defaultdict
from collections import deque
from env_simulator import MakeEnv
from optparse import OptionParser
import os
from policyNet import Actor
from utilis import *
from pathos.multiprocessing import ThreadingPool as Pool

def InitialiseActor(g_1, lr, edges, num_of_zones, num_of_agents, GOALS, T_min, T_max):
	num_of_GOALS = len(np.unique(GOALS))
	with g_1.as_default():
		polNNs = []
		#input_vars = []
		for i in range(0, num_of_zones):
			inp_var = tf.placeholder(shape=[None, num_of_zones+num_of_agents], dtype=tf.float32)
			inp_var1 = tf.placeholder(shape=[None, num_of_GOALS, len(edges[i])], dtype=tf.float32)
			inp_var2 = tf.placeholder(shape=[None, num_of_GOALS, len(edges[i])], dtype=tf.float32)
			#input_vars.append(inp_var) 
			polNN = Actor(bi_n = T_max-T_min, learningRate = lr, num_of_GOALS=num_of_GOALS, input_var=inp_var, input_var1=inp_var1, input_var2=inp_var2, numUnitsPerHLayer=64, neighbours=len(edges[i]), zoneID=str(i))
			polNN.setCompGraph()
			polNNs.append(polNN)
	return polNNs
	
def episodeLoop(PSDD, env, options, all_static_info, sess):
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
	valid_actions_dict = {}
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
	observations = [[] for _ in range(num_of_agents)]
	# store sampled actions i
	actions = [[] for _ in range(num_of_agents)]
	valid_actions = [[] for _ in range(num_of_agents)]

	actionZones = [[] for _ in range(num_of_agents)]
	actionTimes = [[] for _ in range(num_of_agents)]
	nusParas = [[] for _ in range(num_of_agents)]

	# Empirical return for each sampled state-action pair
	rewards = [[] for _ in range(num_of_agents)]
	num_collision = 0


	env.reset()
	for nn in range(num_of_agents):
		PSDD[nn].reset()
	
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
		firstVist = []
		nusPara = []
		entropy = []
		valid_actions_step = []

		for nn in range(0, num_of_agents): 
			all_obs = env.vector_observations[nn]
			agentID = int(all_obs[0])
			zoneID = int(all_obs[1])
			destID = int(all_obs[2])
			timeDest = int(all_obs[3])
			goal = int(all_obs[4])
			
			zoneIDs.append(zoneID)
			goals.append(goal)
			destIDs.append(destID)
			timeDests.append(timeDest)

			possibleneighbourlist = []
			for element in range(5, len(all_obs), 2):
				if all_obs[element] != -1:
					possibleneighbourlist.extend(all_obs[element:element+2])                                                                                                     
			obsOnehot = TransObsOnehot(possibleneighbourlist, num_of_zones) + to_one_hot(agentID, num_of_agents)
			firstVist.append(goal)                           								
			nn_input = np.asarray(obsOnehot).reshape((1, num_of_zones+num_of_agents))
			nn_input1 = np.asarray([0]*num_of_GOALS*len(edges[zoneID])).reshape((1, num_of_GOALS, len(edges[zoneID])))
			obs.append(obsOnehot)
			
			if(destID == -1 and timeDest == -1 and (goal != -1 and goal != zoneID)):   
				valid_literals, visited_nodes_dict = PSDD[nn].get_valid_literal()
				new_set = [PSDD[nn].lit_edge_map[ii] for ii in valid_literals]

				possible_zone_actions = [xx-1 if xx != zoneID+1 else yy-1 for xx, yy in new_set]
				possible_actions = [1 if z_a in possible_zone_actions else 0 for z_a in edges[zoneID] ]

				nn_input2 = np.asarray(possible_actions*num_of_GOALS).reshape((1, num_of_GOALS, len(edges[zoneID])))
				nn_outputs = polNNs[zoneID].getAction(sess, nn_input, nn_input1, nn_input2)                          
				probs = nn_outputs[GOALS_index[goal]][0]   	
				
				a = np.random.choice(len(edges[zoneID]), p=probs)
				for xx in new_set:
					if edges[zoneID][a]+1 in xx:
						edge_selected = xx

				new_e =PSDD[nn].edge_lit_map[edge_selected]
				PSDD[nn].evi_edges.append(new_e)			
				PSDD[nn].nodes_info = visited_nodes_dict[new_e]

				actOneHot = to_one_hot(a, len(edges[zoneID]))
				nn_input1[0,GOALS_index[goal],:] = actOneHot
				nn_outputs = polNNs[zoneID].getAction(sess, nn_input, nn_input1, nn_input2)   
				nu = nn_outputs[GOALS_index[goal]+num_of_GOALS][0][0]                       

				nusPara.append(nu)
				m = T_max - T_min
				timeTodest = T_min + np.random.binomial(m, nu)
				actionIDs.append(a)
				actionZone.append(edges[zoneID][a])
				actionTime.append(timeTodest-T_min)
				action_matrix.append([edges[zoneID][a], timeTodest])
				valid_actions_step.append(possible_actions)
			else:
				actionIDs.append(-1)
				actionZone.append(-1)
				actionTime.append(-1)
				action_matrix.append([-1.0, -1.0])
				nusPara.append(-1)
				entropy.append(-1)
				valid_actions_step.append([-1]*len(edges[zoneID]))

		action_matrix = np.array(action_matrix)      
		env.step(action_matrix)

		#collect states and actions   
		for nn in range(num_of_agents):
			states[nn].append(zoneIDs[nn])
			states_com2[nn].append(destIDs[nn])
			states_com3[nn].append(timeDests[nn])
			states_com4[nn].append(firstVist[nn])
		   
			
			observations[nn].append(obs[nn])
			actions[nn].append(actionIDs[nn])

			actionZones[nn].append(actionZone[nn])
			actionTimes[nn].append(actionTime[nn])
			nusParas[nn].append(nusPara[nn])
			
			valid_actions[nn].append(valid_actions_step[nn])
	   
		#collect rewards
		#penalty for violating collison
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

	resultSavePath = './log/results/'+str(options.mapSize)+'/agents'+str(options.numAgents)+'/'+options.experimentname
	with open(resultSavePath, 'a+') as f:
		f.write(json.dumps(states))
		f.write('\n')

	#Calculate empirical return 
	rets = []
	for nn in range(num_of_agents):
		rewards_nn = rewards[nn]
		rets_nn = []
		return_so_far = 0
		for t in range(len(rewards_nn) - 1, -1, -1):
			return_so_far = rewards_nn[t] + discount * return_so_far
			rets_nn.append(return_so_far)
		# The returns are stored backwards in time, so we need to revert it
		rets_nn = np.array(rets_nn[::-1])
		# normalise returns
		rets_nn = (rets_nn - np.mean(rets_nn)) / (np.std(rets_nn) + 1e-8)
		rets.append(rets_nn)
				 
	# # episode ends, samples for training actor network
	for nn in range(num_of_agents):		
		oneAgent_states = states[nn]
		oneAgent_states4 = states_com4[nn]
		oneAgent_Times = actionTimes[nn]
		oneAgent_observations = observations[nn]
		oneAgent_actions = actions[nn]    
		oneAgent_valid_actions = valid_actions[nn]

		for ss in range(0, len(oneAgent_states)-1):
			state = oneAgent_states[ss]                                   
			goal = oneAgent_states4[ss]
			tToDes = oneAgent_Times[ss]
			act = oneAgent_actions[ss]
			valid_action = oneAgent_valid_actions[ss]		
			if act != -1:                           
				if state in obsdict:
					obsdict[state].append(oneAgent_observations[ss])
					timeTodestdict[state].append(tToDes)

					act_input = [0]*num_of_GOALS*len(edges[state])
					actOneHot = to_one_hot(act, len(edges[state]))
					act_input[GOALS_index[goal]*len(edges[state]): (GOALS_index[goal]+1)*len(edges[state])] = actOneHot
					actdict[state].append(act_input)
					valid_actions_dict[state].append(valid_action*num_of_GOALS)

					retdict[state].append(rets[nn][ss])                              
					nextretdict[state].append(rets[nn][ss+1])					  
				else:
					obsdict[state] = []
					actdict[state] = []
					retdict[state] = []
					timeTodestdict[state] = []
					nextretdict[state] = []  
					valid_actions_dict[state] = []

					obsdict[state].append(oneAgent_observations[ss])
					timeTodestdict[state].append(tToDes)
					
					act_input = [0]*num_of_GOALS*len(edges[state])
					actOneHot = to_one_hot(act, len(edges[state]))
					act_input[GOALS_index[goal]*len(edges[state]): (GOALS_index[goal]+1)*len(edges[state])] = actOneHot
					actdict[state].append(act_input)
					valid_actions_dict[state].append(valid_action*num_of_GOALS)

					retdict[state].append(rets[nn][ss])                          
					nextretdict[state].append(rets[nn][ss+1])
	return (total_len, num_collision, obsdict, actdict, valid_actions_dict, timeTodestdict, retdict, nextretdict)
																					   
def trainModel(options, cap, edges, g_1, polNNs, length, listofenvironments, T_min, T_max, num_of_agents, num_of_zones, GOALS, landmarks, listofPSDDs):
	uniqueGOALS = list(np.unique(GOALS))
	num_of_GOALS = len(uniqueGOALS)
	GOALS_index = {yy : xx for xx, yy in enumerate(uniqueGOALS)}
	with tf.Session(graph=g_1, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('./log/tflogs/'+str(options.mapSize)+'/agents'+str(options.numAgents)+'/'+options.experimentname, sess.graph)
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
			#new iteration
			print("Iteration->",i)
			resultSavePath = './log/results/'+str(options.mapSize)+'/agents'+str(options.numAgents)
			if not os.path.exists(resultSavePath):
				os.makedirs(resultSavePath)
			with open(resultSavePath+'/'+options.experimentname, 'a+') as f:
				f.write('Iteration'+str(i)+'\n')
			
			obsdict = {} 
			actdict = {}
			valid_actions_dict= {}
			timeTodestdict = {}
			retdict = {}
			nextretdict = {}
			
			result = pool.amap(episodeLoop, listofPSDDs, listofenvironments, [options]*len(listofenvironments), [all_static_info]*len(listofenvironments), [sess]*len(listofenvironments))
			all_dictionaries_all_episodes = result.get()

			for data_from_env in range(0,N):  
				total_len_eps, num_collision_eps, obsdict_eps, actdict_eps, valid_actions_dict_eps, timeTodestdict_eps, retdict_eps, nextretdict_eps = all_dictionaries_all_episodes[data_from_env]				
				for x in range(0, num_of_zones):
					if x in obsdict_eps:
						if x in obsdict:
							obsdict[x].extend(obsdict_eps[x])
							actdict[x].extend(actdict_eps[x])
							valid_actions_dict[x].extend(valid_actions_dict_eps[x])
							timeTodestdict[x].extend(timeTodestdict_eps[x])
							retdict[x].extend(retdict_eps[x])
							nextretdict[x].extend(nextretdict_eps[x])
						else:
							obsdict[x] = []
							actdict[x] = []
							timeTodestdict[x] = []
							retdict[x] = []
							nextretdict[x] = []
							valid_actions_dict[x] = []
							obsdict[x].extend(obsdict_eps[x])
							actdict[x].extend(actdict_eps[x])
							valid_actions_dict[x].extend(valid_actions_dict_eps[x])
							timeTodestdict[x].extend(timeTodestdict_eps[x])
							retdict[x].extend(retdict_eps[x])
							nextretdict[x].extend(nextretdict_eps[x])
				paths.append(total_len_eps)
				Collision.append(num_collision_eps)  
			#iteration ends, start training Actor network
			loss2_before = 0
			loss2_after = 0
			for x in range(0, num_of_zones):
				inpdict = {}
				if x in obsdict:
					inpdict[polNNs[x].input_var] = np.asarray(obsdict[x]).reshape((len(obsdict[x]), num_of_zones+num_of_agents))
					inpdict[polNNs[x].input_var1] = np.asarray(actdict[x]).reshape((len(actdict[x]), num_of_GOALS, len(edges[x])))
					inpdict[polNNs[x].act_var] = np.asarray(actdict[x]).reshape((len(actdict[x]), num_of_GOALS, len(edges[x])))
					inpdict[polNNs[x].input_var2] = np.asarray(valid_actions_dict[x]).reshape((-1, num_of_GOALS, len(edges[x])))
					inpdict[polNNs[x].ret_var] = np.asarray(retdict[x]).reshape((1, len(retdict[x])))					
					inpdict[polNNs[x].nextret_var] = np.asarray(nextretdict[x]).reshape((1,len(nextretdict[x])))
					inpdict[polNNs[x].phi] = np.asarray(timeTodestdict[x]).reshape((1,len(timeTodestdict[x])))     
					loss2_before += sess.run(polNNs[x].loss, feed_dict=inpdict)
					sess.run(polNNs[x].learning_step, feed_dict=inpdict)
					loss2_after += sess.run(polNNs[x].loss, feed_dict=inpdict)
			summary = tf.Summary()            
			summary.value.add(tag='summaries/length_of_path', simple_value = np.mean(paths[i*N : (i+1)*N]))
			summary.value.add(tag='summaries/num_collision', simple_value = np.mean(Collision[i*N : (i+1)*N]))
			summary.value.add(tag='summaries/actor_training_loss', simple_value = (loss2_before+loss2_after)/2)  
			writer.add_summary(summary, i)          
			writer.flush()    
		saver = tf.train.Saver()
		save_path = './log/trainedModel/'+str(options.mapSize)+'/agents'+str(options.numAgents)+'/'+options.experimentname
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
	options.landmarkConfigPath = '../../data/landmarkFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'.txt'

	T_min, T_max, max_cap, num_of_zones, num_of_agents, start_zones, goal_zones, STARTS, GOALS, start_goal, landmarks = readConfiguration(options.configPath, options.landmarkConfigPath)
	cap = capInitialise(num_of_agents, num_of_zones, max_cap, start_goal, randomSeed=666)
	length = shortestPath(options.modelPath)
	edges = constructGraph(options.modelPath)
	listofenvironments = envInitialise(options, num_of_agents, STARTS, GOALS, edges)
	listofPSDDs = psddInitialise(options, STARTS, GOALS)

	g_1 = tf.Graph()      
	polNNs = InitialiseActor(g_1, options.lr, edges, num_of_zones, num_of_agents, GOALS, T_min, T_max)
	trainModel(options, cap, edges, g_1, polNNs, length, listofenvironments, T_min, T_max, num_of_agents, num_of_zones, GOALS, landmarks, listofPSDDs)

if __name__ == '__main__':
	main()























