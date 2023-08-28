# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import json

configPath= 'C:/Users/jjling.2018/Desktop/research projects/ZBPF/Experiments/10x10/agents20/10x10_ex'+str(0)+'.txt'    
# configPath= 'C:/Users/jjling.2018/Dropbox/Research Project/PSDD/exp/5x5_l5/config/N6/5x5_ex'+str(0)+'.txt'
f = open(configPath, 'r')
configfile = f.read().split('\n')
T_min = int(configfile[1])
T_max = int(configfile[3])
max_cap = int(configfile[5])
num_of_zones = int(configfile[7])
num_of_agents = int(configfile[9])
start_zones = json.loads(configfile[11])
goal_zones = json.loads(configfile[13])
config = json.loads(configfile[21])
STARTS  = json.loads(configfile[15])
GOALS = json.loads(configfile[17])
start_goal = start_zones + goal_zones
max_capacity = max_cap
cap = {}
np.random.seed(666)
for nn in range(num_of_zones):  
    if nn in start_goal:
        cap[nn] = num_of_agents
    else:
        cap[nn] = np.random.randint(1,1+max_capacity)


modelPath = 'C:/Users/jjling.2018/ml-agents/UnitySDK/Assets/ML-Agents/Examples/MAPF/Models/5x5_test-INPUT.model'     
f = open(modelPath, 'r')
model = f.read().split('\n')

for i in range(0,3):
    with open('NEWtwofloor-INPUT.model', 'a') as f:
        string = model[i]+'\n'
        f.write(string)
        
# =============================================================================
# nn = 0         
# for i in range(3,len(model)-1):
#     with open('NEWtwofloor-INPUT.model', 'a') as f:
#         
#         
#         
#         num_digits = len(str(nn))
#         string = model[i][0:num_digits+1] + str(cap[nn]) + model[i][num_digits+2:-1]+'\n'
#         f.write(string)
# 
#     nn += 1
# =============================================================================
nn = 0 
for i in range(3,len(model)-1):
    with open('NEWtwofloor-INPUT.model', 'a') as f:
        
        line = model[i].split(' ')[:-1]
        
        string = line[0] + ' ' + str(cap[nn]) + ' '
        for j in range(1,len(line)):
            string += line[j]+',1,5 '
        string += '\n'    
        f.write(string)

    nn += 1





