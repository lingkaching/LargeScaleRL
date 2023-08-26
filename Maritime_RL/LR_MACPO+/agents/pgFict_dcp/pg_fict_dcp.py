"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 27 Jun 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print("# ============================ START ============================ #")
# ================================ Imports ================================ #
from random import betavariate
import sys
import os
from pprint import pprint
import time

from numpy.lib.function_base import gradient
import auxLib as ax
import pdb
import rlcompleter
import numpy as np
from parameters import constraintBeta, LOAD_MODEL, TINY, SAVE_MODEL, KEEP_MODELS, MAX_BINARY_LENGTH, LEARNING_RATE, LEARNING_RATE_LAMBDA, OPTIMIZER, BATCH_SIZE, DISCOUNT, SHOULD_LOG, MAP_ID, VF_NORM, NUM_CORES, HORIZON, WEIGHT_VARIANCE, INITIALLAMBDA
from numpy import array
import pdb
import rlcompleter
import torch as tc
import torch
import torch.nn as nn
import networkx as nx
from tensorboardX import SummaryWriter

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# =============================== Variables ================================== #

# ipDim = self.totalZONES #(TOTAL_VESSEL + 1) * self.totalZONES
# h1Dim = self.totalZONES
# h2Dim = self.totalZONES
# opDim = self.totalZONES

# ============================================================================ #
class cost(nn.Module):
    def __init__(self, totalZONES, seed):
        super(cost, self).__init__()
        tc.manual_seed(seed)
        tc.set_num_threads(NUM_CORES)

        self.totalZONES = totalZONES
        self.iDim = 1  # nt(z)
        self.hDim_1 = 2 * self.iDim
        self.hDim_2 = 2 * self.iDim

        self.linear1 = nn.ModuleList()
        self.tanh1 = nn.ModuleList()
        # self.ln1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.tanh2 = nn.ModuleList()
        # self.ln2 = nn.ModuleList()
        self.linear3 = nn.ModuleList()
        self.sigmoid = nn.ModuleList()

        layerIdx = 0
        for z in range(self.totalZONES):
            self.linear1.append(nn.Linear(self.iDim, self.hDim_1, bias=True))
            self.linear1[layerIdx].weight.data.normal_(1, WEIGHT_VARIANCE)
            self.linear1[layerIdx].bias.data.normal_(1, WEIGHT_VARIANCE)
            self.tanh1.append(nn.Tanh())


            self.linear2.append(nn.Linear(self.hDim_1, self.hDim_2, bias=True))
            self.linear2[layerIdx].weight.data.normal_(1, WEIGHT_VARIANCE)
            self.linear2[layerIdx].bias.data.normal_(1, WEIGHT_VARIANCE)
            self.tanh2.append(nn.Tanh())

            self.linear3.append(nn.Linear(self.hDim_2, 1, bias=True))
            self.linear3[layerIdx].weight.data.normal_(1, WEIGHT_VARIANCE)
            self.linear3[layerIdx].bias.data.normal_(1, WEIGHT_VARIANCE)
            self.sigmoid.append(nn.Sigmoid())
            layerIdx += 1


    def forward(self, x, dtPt):

        #nt is a tensor with shape (-1, self.totalZONES)
        nt = tc.tensor(x).float()
        dtPt = nt.shape[0]

        layerIdx = 0
        output = tc.tensor([])

        for z in range(self.totalZONES):
            x = nt[:, z].unsqueeze(1)

            # 1st Layer
            x = self.linear1[layerIdx](x)
            x = self.tanh1[layerIdx](x)

            # 2nd Layer
            x = self.linear2[layerIdx](x)
            x = self.tanh2[layerIdx](x)
           
            # 3rd Layer
            x = self.linear3[layerIdx](x)
            c = self.sigmoid[layerIdx](x)

            layerIdx += 1

            output = tc.cat((output, c), 1)

        return output
    

class actor(nn.Module):
    def __init__(self, totalZONES, planningZONES, zGraph, seed):
        super(actor, self).__init__()

        tc.manual_seed(seed)
        tc.set_num_threads(NUM_CORES)
        self.planningZONES = planningZONES
        self.totalZONES = totalZONES
        self.zGraph = zGraph
        self.iDim = 2  # < nt(z), nt(z')>
        self.hDim_1 = 2 * self.iDim
        self.hDim_2 = 2 * self.iDim

        self.linear1 = nn.ModuleList()
        self.tanh1 = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.tanh2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        self.beta = nn.ModuleList()
        self.sigmoid = nn.ModuleList()

        layerIdx = 0
        for z in self.planningZONES:
             for zp in nx.neighbors(self.zGraph, z):

                self.linear1.append(nn.Linear(self.iDim, self.hDim_1, bias=True))
                self.linear1[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.linear1[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.tanh1.append(nn.Tanh())
                self.ln1.append(nn.LayerNorm(self.hDim_1))

                self.linear2.append(nn.Linear(self.hDim_1, self.hDim_2, bias=True))
                self.linear2[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.linear2[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.tanh2.append(nn.Tanh())
                self.ln2.append(nn.LayerNorm(self.hDim_2))

                self.beta.append(nn.Linear(self.hDim_2, 1, bias=True))
                self.beta[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.beta[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.sigmoid.append(nn.Sigmoid())
                layerIdx += 1






    def forward(self, x, dtPt):


        nt = tc.tensor(x).float()
        dtPt = nt.shape[0]
        layerIdx = 0
        output = torch.ones(dtPt, self.totalZONES, self.totalZONES)


     
           
           
        for z in self.planningZONES:        
            for zp in nx.neighbors(self.zGraph, z):
                x = []
                x.append(nt[:, z])
                x.append(nt[:, zp])

                # 1st Layer
                x = tc.stack(x, 1)
                x = self.linear1[layerIdx](x)
                x = self.tanh1[layerIdx](x)
                # x = self.ln1[layerIdx](x)

                # 2nd Layer
                x = self.linear2[layerIdx](x)
                x = self.tanh2[layerIdx](x)
                # x = self.ln2[layerIdx](x)

                # beta
                x = self.beta[layerIdx](x)
                b = self.sigmoid[layerIdx](x)

      
                layerIdx += 1

      
                output[:,z,zp] = b[:,0]
                

       
        return output


class pg_fict_dcp:

    def __init__(self, data=None, load_model=False, dirName="", FileName="", load_path="", seed=None):

        tc.manual_seed(seed)
        tc.set_num_threads(NUM_CORES)
        self.totalZONES = data.totalZONES
        self.planningZONES = data.planningZONES
        self.dummyZONES = data.dummyZONES
        self.termZONES = data.termZONES
        self.rCap = data.rCap
  
        self.T_min_max = data.T_min_max
        # self.TOTAL_VESSEL = data.total_vessels
        self.zGraph = data.zGraph

        # self.instance = instance
        # self.actor = actor(data.totalZONES, data.Mask)
        self.actor = actor(data.totalZONES, data.planningZONES, data.zGraph, seed)
        self.actor_target = actor(data.totalZONES, data.planningZONES, data.zGraph, seed)
        self.cost = cost(data.totalZONES, seed)

        self.dirName = dirName
        self.FileName = FileName
        writerSavePath = './log/'+self.dirName+'/plots/'+self.FileName
        self.dataSavePath = './log/'+self.dirName+'/data/'+self.FileName

        with open(self.dataSavePath+'/totalRes.txt', 'w') as f:
            f.write('Iteration Value\n')
        with open(self.dataSavePath+'/totalDelay.txt', 'w') as f:
            f.write('Iteration Value\n')
        for i in self.planningZONES:
            fileName = '/planningZONE_'+str(i)+'.txt'          
            with open(self.dataSavePath+fileName, 'w') as f:
                f.write('Iteration Value\n')
            fileName = '/planningZONELambda_'+str(i)+'.txt'
            with open(self.dataSavePath+fileName, 'w') as f:
                f.write('Iteration Value\n')

        if load_model:
            print("-----------------------")
            print("Loading Old Model")
            print("-----------------------")
            self.actor = tc.load("./loadModel" + "/" + load_path + '/model/model.pt')
            self.actor.eval()
            #load cost model

        self.optimizer = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.optimizer_target = tc.optim.Adam(self.actor_target.parameters(), lr=LEARNING_RATE)
        self.optimizer_meta = tc.optim.Adam(self.cost.parameters(), lr=0.01)
        self.create_train_parameters()
        self.writer = SummaryWriter(logdir = writerSavePath)
        self.loss = 0

        i = 0
        self.layer_hash = {}
        for z in range(self.totalZONES):
            for zp in range(self.totalZONES):
                self.layer_hash[(z, zp)] = i
                i += 1

        #initisalise lagrangian multipliers
        self.myLambda = [INITIALLAMBDA] * self.totalZONES
        self.myLambda_old = [INITIALLAMBDA] * self.totalZONES

        self.myXi = [50] * self.totalZONES

        #9_3 420
        # if constraintBeta == 0:
        #     self.myBetaPlanningZONES = [0,0,0,0,0,0,2,2,2,2,6,6,16,6,6,7,6,7,7,1,2,2,2]
        # elif constraintBeta == 1:
        #     self.myBetaPlanningZONES = [0,0,0,0,0,0,4,4,5,5,13,13,32,13,13,14,13,14,14,3,4,4,4]
        # else:
        #     self.myBetaPlanningZONES = [0,0,0,0,0,0,8,9,10,10,27,27,65,26,27,29,27,29,28,7,8,8,9]

        # self.myBeta = [0] * self.totalZONES
        # for i in range(len(self.planningZONES)):
        #     self.myBeta[self.planningZONES[i]] = self.myBetaPlanningZONES[i]

        # self.myBeta = [0] * self.totalZONES

        #9_3 300
        if constraintBeta == 0:
           self.myBetaPlanningZONES = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif constraintBeta == 1:
           self.myBetaPlanningZONES = [0,0,0,0,0,0,0,1,1,1,3,3,11,4,4,4,4,4,4,1,1,1,1]
        elif constraintBeta == 2:
           self.myBetaPlanningZONES = [0,0,0,0,0,0,1,2,2,2,7,7,23,8,8,9,8,8,8,2,2,2,2]
        else:
           self.myBetaPlanningZONES = [0,0,0,0,0,0,5,8,7,6,23,23,70,25,24,27,25,26,26,6,8,7,7]

        self.myBeta = [0] * self.totalZONES
        for i in range(len(self.planningZONES)):
           self.myBeta[self.planningZONES[i]] = self.myBetaPlanningZONES[i]


   

    def create_train_parameters(self):

        self.target = []
        self.nt_z_1hot = []
        self.epoch = 1
        self.meanAds = 0
        self.stdAds = 0
        self.ntz_zt = []
        self.Q = []
        self.dataPt = BATCH_SIZE * (HORIZON)
        # tau' - t - Tmin_zz
        self.tau_p_t_tmin = np.zeros((self.dataPt, self.totalZONES, self.totalZONES, HORIZON + 1))
        tmpDataPt = 0
        for b in range(BATCH_SIZE):
            for t in range(HORIZON):
                for z in range(self.totalZONES):
                    for zp in range(self.totalZONES):
                        tMin_zz = self.T_min_max[z][zp][0]
                        tMax_zz = self.T_min_max[z][zp][1]
                        for tau_p in range(t + tMin_zz, t + tMax_zz + 1):
                            if tau_p < HORIZON:
                                self.tau_p_t_tmin[tmpDataPt][z][zp][tau_p] = tau_p - t - tMin_zz
                tmpDataPt += 1
        self.tau_p_t_tmin = tc.tensor(self.tau_p_t_tmin)
        # Tmax_zz - tau' + t
        self.tmax_taup_t = np.zeros((self.dataPt, self.totalZONES, self.totalZONES, HORIZON + 1))
        tmpDataPt = 0
        for b in range(BATCH_SIZE):
            for t in range(HORIZON):
                for z in range(self.totalZONES):
                    for zp in range(self.totalZONES):
                        tMin_zz = self.T_min_max[z][zp][0]
                        tMax_zz = self.T_min_max[z][zp][1]
                        for tau_p in range(t + tMin_zz, t + tMax_zz + 1):
                            if tau_p < HORIZON:
                                self.tmax_taup_t[tmpDataPt][z][zp][tau_p] = tMax_zz - tau_p + t
                tmpDataPt += 1
        self.tmax_taup_t = tc.tensor(self.tmax_taup_t)


    def getBeta(self, cT, i_episode):

        zCount = cT.nt_z
        zCount = np.reshape(zCount, (1, 1 * self.totalZONES))
        beta_t = self.actor(zCount, zCount.shape[0])

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

        return beta_t.data.numpy()

    def getCost(self, cT, i_episode):

        zCount = cT.nt_z
        zCount = np.reshape(zCount, (1, 1 * self.totalZONES))
        cost_t = self.cost(zCount, zCount.shape[0])

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

        return cost_t.data.numpy()

    def clear(self):

        self.Return = []
        self.Q = []
        self.ntz_zt = []
        self.nt_z_1hot = []
        self.target = []

    def train(self, i_episode):

        # y_target = tc.tensor(np.array(self.target)).float()

        x = np.array(self.nt_z_1hot)
        ntzztau = tc.tensor(np.array(self.ntz_zt))
        Adv = tc.tensor(np.array(self.Q))


        dtPt = x.shape[0]
        beta = self.actor(x, x.shape[0])

        # ---- log(xi^t_zz)
        beta_log =  tc.log(tc.add(beta, TINY))
        beta_log = tc.reshape(beta_log, (dtPt, self.totalZONES, self.totalZONES, 1))

        # ---- log(1 - xi^t_zz)
        ones = tc.ones((dtPt, self.totalZONES, self.totalZONES))
        one_beta = tc.sub(ones, beta)
        one_beta_log = tc.log(tc.add(one_beta, TINY))
        one_beta_log = tc.reshape(one_beta_log, (dtPt, self.totalZONES, self.totalZONES, 1))

        # ----- nt(z, z', tau_p) * [(tau' - t - tmin_zz)*log(beta_t_zz) + (tmax_zz - (tau' - t)) * (1 - log(beta_t_zz))]

        op3 = tc.mul(self.tau_p_t_tmin, beta_log.double())
        op4 = tc.mul(self.tmax_taup_t, one_beta_log.double())
        op5 = tc.add(op3, op4)
        op6 = tc.mul(op5, ntzztau)
        op7 = tc.mul(op6, Adv)


        op8 = tc.sum(op7, 3)
        op8 = op8.reshape(BATCH_SIZE * HORIZON, self.totalZONES * self.totalZONES)
        op8 = op8.transpose_(0, 1)
        op8 = op8.reshape(self.totalZONES, self.totalZONES, BATCH_SIZE * HORIZON)

        # op7 = tc.reshape(op7, (BATCH_SIZE*HORIZON, self.totalZONES * self.totalZONES * (HORIZON+1)))
        # op8 = tc.sum(op7, 1)
        # loss = -tc.mean(op8)
        loss = tc.sum(-tc.mean(op8, 2))
        self.loss = float(loss.data)
        self.optimizer.zero_grad()
        loss.backward()


            
        # pg_loss_grad = {}
        # for z in self.planningZONES:
        #     for zp in nx.neighbors(self.zGraph, z):
        #         pg_loss_grad[(z, zp)] = []
        #         indx = self.layer_hash[(z, zp)]
        #         pg_loss_grad[(z, zp)].append(self.actor.linear1[indx].weight.grad)     
        #         pg_loss_grad[(z, zp)].append(self.actor.linear1[indx].bias.grad)     
        #         pg_loss_grad[(z, zp)].append(self.actor.linear2[indx].weight.grad)     
        #         pg_loss_grad[(z, zp)].append(self.actor.linear2[indx].bias.grad)  
        #         pg_loss_grad[(z, zp)].append(self.actor.beta[indx].weight.grad)     
        #         pg_loss_grad[(z, zp)].append(self.actor.beta[indx].bias.grad)  

        # pg_loss_grad_norm = {}
        # for z in self.planningZONES:
        #     for zp in nx.neighbors(self.zGraph, z):
        #         grad_norm = 0
        #         for grad in pg_loss_grad[(z, zp)]:
        #             # print(grad1, grad2)
        #             grad_norm += tc.norm(grad)

        #         pg_loss_grad_norm[(z,zp)] = grad_norm / len(pg_loss_grad[(z, zp)])


        self.optimizer.step()

        return None


    def trainMeta(self, i_episode, buffer_nt_z, buffer_ntz_zt):
        #compute pg_loss1 w.r.t new policy parameters
        x = np.array(self.nt_z_1hot)
        ntzztau = tc.tensor(np.array(self.ntz_zt))
        Adv = tc.tensor(np.array(self.Q))

        dtPt = x.shape[0]
        beta = self.actor(x, x.shape[0])

        # ---- log(xi^t_zz)
        beta_log =  tc.log(tc.add(beta, TINY))
        beta_log = tc.reshape(beta_log, (dtPt, self.totalZONES, self.totalZONES, 1))

        # ---- log(1 - xi^t_zz)
        ones = tc.ones((dtPt, self.totalZONES, self.totalZONES))
        one_beta = tc.sub(ones, beta)
        one_beta_log = tc.log(tc.add(one_beta, TINY))
        one_beta_log = tc.reshape(one_beta_log, (dtPt, self.totalZONES, self.totalZONES, 1))

        # ----- nt(z, z', tau_p) * [(tau' - t - tmin_zz)*log(beta_t_zz) + (tmax_zz - (tau' - t)) * (1 - log(beta_t_zz))]

        op3 = tc.mul(self.tau_p_t_tmin, beta_log.double())
        op4 = tc.mul(self.tmax_taup_t, one_beta_log.double())
        op5 = tc.add(op3, op4)
        op6 = tc.mul(op5, ntzztau)
        # ----- nt(z, z', tau_p) * [(tau' - t - tmin_zz)*log(beta_t_zz) + (tmax_zz - (tau' - t)) * (1 - log(beta_t_zz))] * adv
        op7 = tc.mul(op6, Adv)

        op8 = tc.sum(op7, 3)
        op8 = op8.reshape(BATCH_SIZE * HORIZON, self.totalZONES * self.totalZONES)
        op8 = op8.transpose_(0, 1)
        op8 = op8.reshape(self.totalZONES, self.totalZONES, BATCH_SIZE * HORIZON)

        pg_loss1 = tc.sum(tc.mean(op8, 2))
        

        self.optimizer.zero_grad()
        pg_loss1.backward()
        
        pg_loss1_grad = {}
        for z in self.planningZONES:
            for zp in nx.neighbors(self.zGraph, z):
                pg_loss1_grad[(z, zp)] = []
                indx = self.layer_hash[(z, zp)]
                pg_loss1_grad[(z, zp)].append(self.actor.linear1[indx].weight.grad)     
                pg_loss1_grad[(z, zp)].append(self.actor.linear1[indx].bias.grad)     
                pg_loss1_grad[(z, zp)].append(self.actor.linear2[indx].weight.grad)     
                pg_loss1_grad[(z, zp)].append(self.actor.linear2[indx].bias.grad)  
                pg_loss1_grad[(z, zp)].append(self.actor.beta[indx].weight.grad)     
                pg_loss1_grad[(z, zp)].append(self.actor.beta[indx].bias.grad)  

        # print(pg_loss1_grad)

        #compute pg_loss2
        beta = self.actor_target(x, x.shape[0])

        # ---- log(xi^t_zz)
        beta_log =  tc.log(tc.add(beta, TINY))
        beta_log = tc.reshape(beta_log, (dtPt, self.totalZONES, self.totalZONES, 1))

        # ---- log(1 - xi^t_zz)
        ones = tc.ones((dtPt, self.totalZONES, self.totalZONES))
        one_beta = tc.sub(ones, beta)
        one_beta_log = tc.log(tc.add(one_beta, TINY))
        one_beta_log = tc.reshape(one_beta_log, (dtPt, self.totalZONES, self.totalZONES, 1))

        # ----- [(tau' - t - tmin_zz)*log(beta_t_zz) + (tmax_zz - (tau' - t)) * (1 - log(beta_t_zz))]

        op3 = tc.mul(self.tau_p_t_tmin, beta_log.double())
        op4 = tc.mul(self.tmax_taup_t, one_beta_log.double())
        op7 = tc.add(op3, op4)
        # op7 = tc.mul(op5, ntzztau)
     

        op8 = tc.sum(op7, 3)
        op8 = op8.reshape(BATCH_SIZE * HORIZON, self.totalZONES * self.totalZONES)
        op8 = op8.transpose_(0, 1)
        op8 = op8.reshape(self.totalZONES, self.totalZONES, BATCH_SIZE * HORIZON)

        pg_loss2 = tc.sum(tc.mean(op8, 2))

        self.optimizer_target.zero_grad()
        pg_loss2.backward()
        
        pg_loss2_grad = {}
        for z in self.planningZONES:
            for zp in nx.neighbors(self.zGraph, z):
                pg_loss2_grad[(z, zp)] = []
                indx = self.layer_hash[(z, zp)]
                pg_loss2_grad[(z, zp)].append(self.actor_target.linear1[indx].weight.grad)     
                pg_loss2_grad[(z, zp)].append(self.actor_target.linear1[indx].bias.grad)     
                pg_loss2_grad[(z, zp)].append(self.actor_target.linear2[indx].weight.grad)     
                pg_loss2_grad[(z, zp)].append(self.actor_target.linear2[indx].bias.grad)  
                pg_loss2_grad[(z, zp)].append(self.actor_target.beta[indx].weight.grad)     
                pg_loss2_grad[(z, zp)].append(self.actor_target.beta[indx].bias.grad)  
        
        # print(pg_loss2_grad)


        grad_total = {}
        for z in self.planningZONES:
            for zp in nx.neighbors(self.zGraph, z):
                grad_total_z_zp = 0

                for grad1, grad2 in zip(pg_loss1_grad[(z, zp)], pg_loss2_grad[(z, zp)]):
                    # print(grad1, grad2)
                    grad_total_z_zp += tc.sum(grad1 * grad2)

                grad_total[(z,zp)] = grad_total_z_zp

        # print(grad_total)

        buffer_nt_z = tc.tensor(buffer_nt_z,dtype=torch.float32)
        buffer_ntz_zt = tc.tensor(buffer_ntz_zt, dtype=torch.float32)
        #compute cummulative cost for updateing meta parameters
        learnedUnit = self.cost(x, x.shape[0])
        rCap = tc.tensor(self.rCap)

        learnedCost = tc.relu(tc.sub(learnedUnit, tc.div(rCap, tc.add(buffer_nt_z, TINY))))
        learnedCostTotal = tc.relu(tc.sub(tc.mul(learnedUnit, buffer_nt_z), rCap))
        actualCost = tc.relu(tc.sub(buffer_nt_z, rCap))



        # rets_z = [[] for _ in range(self.totalZONES)]
        # for z in self.planningZONES:
        #     rets = tc.zeros(HORIZON)
        #     return_so_far = torch.tensor(0)
        #     for t in range(len(learnedCost[:,z]) - 1, -1, -1):
        #         return_so_far = learnedCost[:,z][t] + DISCOUNT * return_so_far
        #         rets[t] = return_so_far
        #     # The returns are stored backwards in time, so we need to revert it
        #     # rets = np.array(rets[::-1])
        #     rets_z[z] = rets
        #     # print(rets)
        #     # # normalise returns
        #     # rets = (rets - np.mean(rets)) / (np.std(rets) + 1e-8)
 


        
        # # ------------ Advantage ------------ #
        # q_tmp = tc.zeros((HORIZON, self.totalZONES, self.totalZONES, HORIZON + 1))
        # for t in range(HORIZON):
        #     for z in self.planningZONES:
        #         for z_p in nx.neighbors(self.zGraph, z):
        #             tmin = self.T_min_max[z][z_p][0]
        #             tmax = self.T_min_max[z][z_p][1]
        #             for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON + 1)):
        #                 q_tmp[t][z][z_p][tau_p] = rets_z[z][t]
        #                 # Adv[t][z][z_p][tau_p] = rets[t]


        
        #-------------------------------- #
        R_z_t_zp_tp = tc.zeros(self.totalZONES, HORIZON, self.totalZONES, HORIZON + 1)
        for t in range(HORIZON):
            for z in self.planningZONES:
                tau = t
                for zp in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][zp][0]
                    tmax = self.T_min_max[z][zp][1]
                    for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON+1)):
                        R_z_t_zp_tp[z][tau][zp][tau_p] = tc.sum(learnedCost[t:tau_p+1, z])
 
        q_tmp = tc.zeros(self.totalZONES, HORIZON, self.totalZONES, HORIZON + 1)
        self.V = tc.zeros(self.totalZONES, HORIZON + 1)
        for tau in range(HORIZON - 1, -1, -1):
            for z in self.planningZONES:
                for z_p in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][z_p][0]
                    tmax = self.T_min_max[z][z_p][1]
                    for tau_p in range(tau + tmin, min(tau + tmax + 1, HORIZON + 1)):
                        q_tmp[z][tau][z_p][tau_p] = R_z_t_zp_tp[z][tau][z_p][tau_p] + DISCOUNT * self.V[z_p][tau_p]
                # ------- V -------- #
                self.V[z][tau] = tc.div(tc.sum(q_tmp[z][tau][:][:] * buffer_ntz_zt[tau][z][:][:]), tc.add(
                    tc.sum(buffer_ntz_zt[tau][z]), TINY))
        # print(q_tmp)


        q_tmp = q_tmp.transpose_(0, 1)

        # qValues = tc.sum(qValues, 3)
        qValues = tc.mul(q_tmp, ntzztau)
        qValues = tc.sum(qValues, 3)
        qValues = qValues.reshape(BATCH_SIZE * HORIZON, self.totalZONES * self.totalZONES)
        qValues = qValues.transpose_(0, 1)
        qValues = qValues.reshape(self.totalZONES, self.totalZONES, BATCH_SIZE * HORIZON)
        qValues = tc.mean(qValues, 2)

        
        # loss_meta = 0
        # for z in self.planningZONES:
        #     loss_tmp = 0
        #     for zp in nx.neighbors(self.zGraph, z):
        #         # print(qValues[z][zp])
        #         # print( grad_total[(z,zp)])
        #         loss_tmp += qValues[z][zp] * grad_total[(z,zp)] * self.myLambda[z]
            
                
        #     # loss_meta += - loss_tmp + self.myXi[z] * tc.mean(tc.square(learnedCount[:,z] - buffer_nt_z[:,z]) ) 
        #     # loss_meta += - loss_tmp  + 0 * tc.mean(tc.square(learnedCost[:,z] - actualCost[:,z]) ) 
        #     # loss_meta += tc.mean(tc.square(learnedCost[:,z] - actualCost[:,z]) ) 
        #     loss_meta += - loss_tmp + 75 * tc.mean(tc.square(learnedCostTotal[:,z] - actualCost[:,z])) 

        loss_1 = qValues[2][3] * grad_total[(2,3)] * self.myLambda_old[2]
        loss_2 = tc.mean(tc.square(learnedCostTotal[:,2] - actualCost[:,2])) 

        loss_meta = - qValues[2][3] * grad_total[(2,3)] * self.myLambda_old[2] + self.myXi[2] * tc.mean(tc.square(learnedCostTotal[:,2] - actualCost[:,2])) 
        # loss_meta = self.myXi[2] * tc.mean(tc.square(learnedCostTotal[:,2] - actualCost[:,2])) 


        self.optimizer_meta.zero_grad()
        loss_meta.backward()

        # test_grad = {}
        # for z in self.planningZONES:
        #     test_grad[z] = []
         
        #     test_grad[z].append(self.cost.linear1[z].weight.grad)     
        #     test_grad[z].append(self.cost.linear1[z].bias.grad)     
        #     test_grad[z].append(self.cost.linear2[z].weight.grad)     
        #     test_grad[z].append(self.cost.linear2[z].bias.grad)  
        #     test_grad[z].append(self.cost.linear3[z].weight.grad)     
        #     test_grad[z].append(self.cost.linear3[z].bias.grad)  
        
        # # print(test_grad)


        self.optimizer_meta.step()

        #update lambda
        for z in range(self.totalZONES):
            self.myLambda_old[z] = self.myLambda[z]

        self.myXi[2] = self.myXi[2] * 0.9996

        return loss_meta, loss_1, loss_2


    def storeRollouts(self, buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_ct_z, buffer_beta, buffer_nt_ztz, buffer_rt):

        for t in range(HORIZON):
            zCount_1hot = buffer_nt_z[t]
            zCount_1hot = np.reshape(zCount_1hot, (1, 1 * self.totalZONES))
            self.nt_z_1hot.append(zCount_1hot[0])

        self.ntz_zt.extend(buffer_ntz_zt)


        # ------------compute empirical return----------#
        cost_z = [[] for _ in range(self.totalZONES)]
        for z in self.planningZONES:
            cost = []
            cost_so_far = 0
            for t in range(len(buffer_ct_z[:,z]) - 1, -1, -1):
                cost_so_far = buffer_ct_z[:,z][t] + DISCOUNT * cost_so_far
                cost.append(cost_so_far)
            # The returns are stored backwards in time, so we need to revert it
            cost = np.array(cost[::-1])
            cost_z[z] = cost
            # print(rets)
            # # normalise returns
            # rets = (rets - np.mean(rets)) / (np.std(rets) + 1e-8)
        # print(rets_z)


        
        # ------------ Cost Advantage ------------ #
        Adv1 = np.zeros((HORIZON, self.totalZONES, self.totalZONES, HORIZON + 1))
        for t in range(HORIZON):
            for z in self.planningZONES:
                for z_p in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][z_p][0]
                    tmax = self.T_min_max[z][z_p][1]
                    for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON + 1)):
                        Adv1[t][z][z_p][tau_p] = cost_z[z][t]
                        # Adv[t][z][z_p][tau_p] = rets[t]


       # --------------reward--------- #
        R_z_t_zp_tp = np.zeros((self.totalZONES, HORIZON, self.totalZONES, HORIZON + 1))
        for t in range(HORIZON):
            for z in self.planningZONES:
                tau = t
                for zp in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][zp][0]
                    tmax = self.T_min_max[z][zp][1]
                    for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON+1)):
                        R_z_t_zp_tp[z][tau][zp][tau_p] = np.sum(buffer_rt_z[t:tau_p+1, z])

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

        # ---------- Compute Q & V - Values ----------- #
        # ------- Q ------ #
        q_tmp = np.zeros((self.totalZONES, HORIZON, self.totalZONES, HORIZON + 1))
        self.V = np.zeros((self.totalZONES, HORIZON + 1))
        for tau in range(HORIZON - 1, -1, -1):
            for z in self.planningZONES:
                for z_p in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][z_p][0]
                    tmax = self.T_min_max[z][z_p][1]
                    for tau_p in range(tau + tmin, min(tau + tmax + 1, HORIZON + 1)):
                        q_tmp[z][tau][z_p][tau_p] = R_z_t_zp_tp[z][tau][z_p][tau_p] + DISCOUNT * self.V[z_p][tau_p]
                # ------- V -------- #
                self.V[z][tau] = np.sum(q_tmp[z][tau][:][:] * buffer_ntz_zt[tau][z][:][:]) / np.max(
                    [np.sum(buffer_ntz_zt[tau][z]), TINY])

        #normalization
        # buffer_ntz_zt = np.swapaxes(buffer_ntz_zt, 0, 1)
        # if self.epoch == 1:
        #     self.meanAds = np.sum(q_tmp * buffer_ntz_zt) / HORIZON
        #     self.stdAds = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buffer_ntz_zt) / (HORIZON))

        # else:
        #     self.meanAds1 = np.sum(q_tmp * buffer_ntz_zt) / (HORIZON)
        #     try:
        #         self.stdAds1 = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buffer_ntz_zt ) / (HORIZON))
        #     except RuntimeWarning as e:
        #         print('error found:', e)
        #     self.meanAds = 0.9 * self.meanAds1 + 0.1 * self.meanAds
        #     self.stdAds = 0.9 * self.stdAds1 + 0.1 * self.stdAds
        # q_tmp = (q_tmp - self.meanAds)/(self.stdAds + TINY)
        # self.V = (self.V - self.meanAds)/(self.stdAds + TINY)


        # # ------------ Advantage ------------ #
        # Adv = np.zeros((HORIZON, self.totalZONES, self.totalZONES, HORIZON + 1))
        # for t in range(HORIZON):
        #     for z in self.planningZONES:
        #         for z_p in nx.neighbors(self.zGraph, z):
        #             tmin = self.T_min_max[z][z_p][0]
        #             tmax = self.T_min_max[z][z_p][1]
        #             for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON + 1)):
        #                 Adv[t][z][z_p][tau_p] = q_tmp[z][t][z_p][tau_p]
        #                 Adv[t][z][z_p][tau_p] -= self.V[z][t]

        # self.Q.extend(Adv)

        q_tmp = np.transpose(q_tmp, (1, 0, 2, 3))
        self.Q.extend(Adv1+q_tmp)


    #Write your code here to update lambda
    #what is the constraint
    def trainLambda(self, i_episode, rCap):


        batchSize = int(len(self.nt_z_1hot) / HORIZON)
        batchSamples = np.asarray(self.nt_z_1hot).reshape((batchSize, HORIZON, -1))
        # print(np.asarray(self.nt_z_1hot))
        # print(learnedUnit)
        # print(np.asarray(self.nt_z_1hot)*learnedUnit)
        # print(batchSamples.shape)
        for z in self.planningZONES:
            #compute gradient
            cumCount = 0
            for i in range(batchSize):
                #average over whole horizon
                cumCount += np.sum(np.maximum(np.zeros(HORIZON),batchSamples[i][:,z]-rCap[z]))
                #average over non-zero
                # cumCount += np.sum(batchSamples[i][:,z]) / np.sum(batchSamples[i][:,z] != 0 )
                #max
                # cumCount += np.max(batchSamples[i][:,z])
            avgCumCount = cumCount / batchSize

            #below is the gradient
            gradientZ = avgCumCount - self.myBeta[z]
            #perform gradient ascent and projection
            self.myLambda[z] = max(0, self.myLambda[z] + LEARNING_RATE_LAMBDA * gradientZ)
            
            # print('gradient', z, gradientZ)
            # print("updated lambda:", z, self.myLambda[z])

        # return   

    def update_target(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

    def logStat(self, i_episode, vio, delay, myLambda, pg_loss_grad_norm):
        self.writer.add_scalar('Total ResVio', np.sum(vio), i_episode)
        self.writer.add_scalar('Total Total Delay', delay, i_episode)
               
        for z in range(self.totalZONES):
            if z in self.planningZONES:
                self.writer.add_scalar("Lambda/"+str(z), myLambda[z], i_episode)
                self.writer.add_scalar("ResVio/"+str(z), np.sum(vio[:,z]), i_episode)

             
        with open(self.dataSavePath+'/totalRes.txt', 'a+') as f:
            f.write(str(i_episode)+' '+str(np.sum(vio))+'\n')
        with open(self.dataSavePath+'/totalDelay.txt', 'a+') as f:
            f.write(str(i_episode)+' '+str(delay)+'\n')
        for z in self.planningZONES:
            fileName = '/planningZONE_'+str(z)+'.txt'
            with open(self.dataSavePath+fileName, 'a+') as f:
                f.write(str(i_episode)+' '+str(np.sum(vio[:,z]))+'\n')  

            fileName = '/planningZONELambda_'+str(z)+'.txt'
            with open(self.dataSavePath+fileName, 'a+') as f:
                f.write(str(i_episode)+' '+str(myLambda[z])+'\n')  


             




    def log(self, i_episode, epReward, betaAvg2, vio, delay, myLambda):

        self.writer.add_scalar('Total Rewards', epReward, i_episode)
        self.writer.add_scalar('Total ResVio', vio, i_episode)
        self.writer.add_scalar('Total Delay', delay, i_episode)

        # self.writer.add_scalar('Sample Avg. Rewards', sampleAvg, i_episode)
        # self.writer.add_scalar('Loss', self.loss, i_episode)
        for z in range(self.totalZONES):
            if z in self.planningZONES:
                self.writer.add_scalar("Lambda/"+str(z), myLambda[z], i_episode)

            if (z not in self.dummyZONES) and (z not in self.termZONES):
                for zp in nx.neighbors(self.zGraph, z):
                    # pdb.set_trace()
                    self.writer.add_scalar("Beta/"+str(z)+"_"+str(zp), betaAvg2[z][zp], i_episode)

    def logTest(self, i_episode_test, epReward, vio, delay):

        self.writer.add_scalar('Test Total Rewards', epReward, i_episode_test)
        self.writer.add_scalar('Test Total ResVio', vio, i_episode_test)
        self.writer.add_scalar('Test Total Delay', delay, i_episode_test)
         

    def save_model(self):

        tc.save(self.actor, 'log/'+self.dirName+'/model/'+self.dirName+'.pt')

    def ep_init(self):
        return

# =============================================================================== #
