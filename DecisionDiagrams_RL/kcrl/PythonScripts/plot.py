# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:23:27 2019

@author: jjling.2018
"""
import json
from optparse import OptionParser
import os
from matplotlib import pyplot as plt
import numpy as np
from itertools import groupby
import seaborn as sns; sns.set()







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



if __name__ == '__main__':
    main()




