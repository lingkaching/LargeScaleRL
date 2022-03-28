# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:23:27 2019

@author: jjling.2018
"""
import json
import tensorflow as tf
import networkx as nx 
import numpy as np
import sys
import random
import time
import os
from env_simulator import MakeEnv
from knowledgeCompilation import A_s

def psddInitialise(options, STARTS, GOALS):

	node_id_list_file = '../../data/jsonFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_node_id_list.json'
	node_type_list_file = '../../data/jsonFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_node_type_list.json'
	elements_list_file = '../../data/jsonFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_elements_list.json'
	literal_list_file = '../../data/jsonFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_literal_list.json'
	vtree_id_list_file =  '../../data/jsonFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_vtree_id_list.json'
	vtree_ancessors_file = '../../data/jsonFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_vtree_ancessors.json'
	edge_var_file = '../../data/sddFiles/'+str(options.mapSize)+'/'+'edge_var_map_'+str(options.mapSize)+'.txt'
	json_file = '../../data/sddFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'.json'
	lit_edge_file = '../../data/sddFiles/'+str(options.mapSize)+'/'+'lit_edge_map_'+str(options.mapSize)+'_zdd.txt'

	listofas = []
	for _ in range(0, options.N):
		As = {}
		agent_id = 0
		mapSize = int(options.mapSize.split('x')[0])
		for s,g in zip(STARTS, GOALS):
			if s < mapSize:
				s_super = (s+1, s+1+int(mapSize*mapSize))
			else:
				s_super = (s+1, s+1+int(2*mapSize))
			if g < mapSize:
				d_super = (g+1, g+1+int(mapSize*mapSize))
			else:
				d_super = (g+1, g+1+int(2*mapSize))
			# num_psdd_node, num_vtree_node = 20279+1, 427+1
			As[agent_id] = A_s(node_id_list_file, node_type_list_file, elements_list_file, literal_list_file, vtree_id_list_file, vtree_ancessors_file, edge_var_file, json_file, lit_edge_file, s_super, d_super)
			agent_id += 1 
		listofas.append(As)
	return listofas

def envInitialise(options, num_of_agents, STARTS, GOALS, edges):
	listofenvironments = []
	for _ in range(0, options.N):
		env = MakeEnv(edges, num_of_agents, STARTS, GOALS) 
		listofenvironments.append(env)
	return listofenvironments

def capInitialise(num_of_agents, num_of_zones, max_cap, start_goal, randomSeed):
	cap = {}
	#set random seed here
	np.random.seed(randomSeed)
	for nn in range(num_of_zones):  
		if nn in start_goal:
			cap[nn] = num_of_agents
		else:
			cap[nn] = np.random.randint(1,1+max_cap)
	return cap

def readConfiguration(configPath, landmarkConfigPath):
	#read landmark file
	f = open(landmarkConfigPath, 'r')
	configfile = f.read().split('\n')
	landmarks = json.loads(configfile[1])
	landmarks = [l-1 for l in landmarks]

	#read configuration file
	f = open(configPath, 'r')
	configfile = f.read().split('\n')
	T_min = int(configfile[1])
	T_max = int(configfile[3])
	max_cap = int(configfile[5])
	num_of_zones = int(configfile[7])
	num_of_agents = int(configfile[9])
	start_zones = json.loads(configfile[11])
	goal_zones = json.loads(configfile[13])
	STARTS = json.loads(configfile[15])
	GOALS = json.loads(configfile[17])
	start_goal = start_zones + goal_zones
	return T_min, T_max, max_cap, num_of_zones, num_of_agents, start_zones, goal_zones, STARTS, GOALS, start_goal, landmarks

def shortestPath(modelPath):
	f = open(modelPath, 'r')
	model = f.read().split('\n')
	#all pair shortest path
	edges = []
	for i in range(3, len(model)):
		edgeRecords = model[i].split(' ')[:-1]
		for j in range(1, len(edgeRecords)):
			edges.append((edgeRecords[0], edgeRecords[j]))  
	#create directed graph
	G = nx.DiGraph()
	G.add_edges_from(edges)    
	length = dict(nx.all_pairs_shortest_path_length(G))
	return length

def constructGraph(modelPath):
	f = open(modelPath, 'r')
	model = f.read().split('\n')
	num_of_zones = int(model[0].strip().split(':')[-1].strip())
	edges = {}
	for i in range(3, len(model)):
		if(len(model[i]) == 0):
			continue
		edgeRecords = model[i].strip().split(' ')
		edges[int(edgeRecords[0])] = [int(x) for x in edgeRecords[1:]]      
	return edges
	
def to_one_hot(ID, N):
	arr = [0]*N
	if ID != -1:
		arr[ID] = 1
	return arr  
																							
def TransObsOnehot(possibleneighbourlist, N):
	arr = [0]*N
	for i in range(0, len(possibleneighbourlist), 2):
		ID = int(possibleneighbourlist[i])
		num_agents = int(possibleneighbourlist[i+1])
		arr[ID] = num_agents
	return arr  

def potential(length, currentZ, desZ, time, goal):
	if desZ == -1:
		return -length[str(currentZ)][str(goal)]    
	else:
		return -length[str(desZ)][str(goal)] - time  











