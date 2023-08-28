# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:32:13 2020

@author: jjling.2018
"""
import json
from itertools import groupby
import numpy as np
from mlagents.envs import UnityEnvironment

def extractPolicy(Paths, num_of_agents, GOALS):
    Paths_new = []
    for nn in range(num_of_agents):
    	path = Paths[nn]
    	goal_agent = GOALS[nn]
    	path = [x for x, y in groupby(path)] 
    	Paths_new.append(path)

    max_len = 0
    for nn in range(num_of_agents):
        if len(Paths_new[nn]) > max_len:
            max_len = len(Paths_new[nn])

    for nn in range(num_of_agents):
        diff = max_len - len(Paths_new[nn]) 
        Paths_new[nn].extend([Paths_new[nn][-1]]*diff)
    Paths = Paths_new

    Steps = [[] for _ in range(num_of_agents)]
    for nn in range(num_of_agents):
        for x,y in groupby(Paths[nn]):
            # Steps[nn].append([x,5])
            Steps[nn].append([x,len(list(y))])
    
    Policies = [[] for _ in range(num_of_agents)]
    for nn in range(num_of_agents):
       for i in range(len(Steps[nn])-1):
           Policies[nn].append([Steps[nn][i+1][0],Steps[nn][i][1]])
    return Policies

def readPath(pathFile):
    f = open(pathFile, 'r')
    simulatedPath = {}
    i = 0
    for line in f:
        if line.startswith('I'): 
            simulatedPath[i] = []
            i += 1
        else:
            try:
                path = json.loads(line)
            except ValueError:
                continue
            simulatedPath[i-1].append(json.loads(line))
    return simulatedPath



def readModel(modelPath):
    f = open(modelPath, 'r')
    model = f.read().split('\n')
        # num_of_zones = int(model[0].strip().split(':')[-1].strip())
    edges = {}
    for i in range(3, len(model)):
        if(len(model[i]) == 0):
            continue
        edgeRecords = model[i].strip().split(' ')
        edges[int(edgeRecords[0])] = [int(x) for x in edgeRecords[1:]]      
    return edges



def simulation(env, simulatedPath, num_of_agents, GOALS):
    for ite in range(1,100):
        default_brain = env.brain_names[0]
        brain = env.brains[default_brain]
        # env_info = env.reset(train_mode=False, config={"ReadConfig#0F#1T" : 1.0, "InstanceID" : 1})[default_brain] 
        env_info = env.reset(train_mode=False, config={"ReadConfig#0F#1T" : 1.0})[default_brain] 
        action_matrix = np.array([[-1,-1] for _ in range(num_of_agents)])             
        env_info = env.step([action_matrix])[default_brain]     
        flags = [False] * num_of_agents

        Policies = extractPolicy(simulatedPath[ite][0], num_of_agents, GOALS)

        t = 0
        episideEndFlag = False 
        while t < 500:
            action_matrix = []
            for nn in range(0, num_of_agents): 
                all_obs = env_info.vector_observations[nn]
                agentID = int(all_obs[0])
                zoneID = int(all_obs[1])
                destID = int(all_obs[2])
                timeDest = int(all_obs[3])
                goal = int(all_obs[4])
                if(destID == -1 and timeDest == -1 and (goal != -1 and goal != zoneID)): 
                    if len(Policies[nn]) == 0:
                        episideEndFlag = True
                        break
                    actions = Policies[nn].pop(0)
                    # dis_action = np.random.choice(edges[zoneID])
                    # con_action = np.random.choice([1,2,3,4,5])
                    # actions = [dis_action, con_action]
                    action_matrix.append(actions)
                else:
                    action_matrix.append([-1.0, -1.0])
            if episideEndFlag:
                break
            else:
                action_matrix = np.array(action_matrix) 
                env_info = env.step(action_matrix)[default_brain]
                
                for nn in range(0, num_of_agents): 
                    if env_info.local_done[nn] == True:		
                        flags[nn] = True
                t += 1
                if all(flags):
                    break

    env.close()


def main():
    modelPath = '../../Models/5x5.model'
    env_name = '../ExecutiveEnv/5x5/Unity Environment'
    pathFile = '../PathFile/5_6_l5_DC_ex0_run0.txt'
 
    num_of_agents = 6    
    GOALS = [23, 22, 23, 2, 2, 1]
    edges = readModel(modelPath)
    simulatedPath = readPath(pathFile)
    env = UnityEnvironment(file_name=env_name, worker_id = 0, seed =1)
    simulation(env, simulatedPath, num_of_agents, GOALS)


if __name__ == '__main__':
	main()





