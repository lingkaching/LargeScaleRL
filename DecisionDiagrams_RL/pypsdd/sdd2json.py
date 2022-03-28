# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:13:24 2020

@author: jjling.2018
"""

from __future__ import print_function
from pypsdd import Vtree,SddManager,PSddManager,io
from pypsdd import DataSet,Prior
from pypsdd import Inst,InstMap
import json
from optparse import OptionParser
import os

def sdd2jsons(vtree_filename, sdd_filename, lit_edge_file, save_name):
    vtree = Vtree.read(vtree_filename)
    smanager = SddManager(vtree)
    io.my_sdd_read(sdd_filename, smanager, save_name)

    vtree_ancessors = {}
    for v in vtree.__iter__():
        if v.var:
            vtree_ancessors[v.var] = [v.id]
            iter_v = v 
            while iter_v.parent != None:
                iter_v = iter_v.parent
                vtree_ancessors[v.var].append(iter_v.id)

    save_path = save_name+'vtree_ancessors.json'
    with open(save_path, 'w') as f:
        json.dump(vtree_ancessors, f)
    return

def psdd2jsons(json_file, vtree_filename, psdd_filename, edge_var_file, save_name):
    vtree = Vtree.read(vtree_filename)
    pmanager1 = PSddManager(vtree)
    psdd = io.psdd_yitao_read(psdd_filename, pmanager1, save_name) #psdd = io.psdd_yitao_read(psdd_filename, pmanager1) #### use io.psdd_jason_read(filename, pmanager)
    Prior.random_parameters(psdd)

    vtree_ancessors = {}
    for v in vtree.__iter__():
        if v.var:
            vtree_ancessors[v.var] = [v.id]
            iter_v = v 
            while iter_v.parent != None:
                iter_v = iter_v.parent
                vtree_ancessors[v.var].append(iter_v.id)

    save_path = save_name+'vtree_ancessors.json'
    with open(save_path, 'w') as f:
        json.dump(vtree_ancessors, f)
    return

def main():
    parser = OptionParser()
    parser.add_option("-f", "--file", type='str', dest='fileType', default='sdd')
    parser.add_option("-s", "--size", type='str', dest='mapSize', default='5x5')
    parser.add_option("-m", "--map", type='str', dest='mapType', default='open', help='either open or landmark')
    parser.add_option("-n", "--numLandmarks", type='int', dest='numLandmarks', default=0)
    (options, args) = parser.parse_args()  
    if options.fileType == 'sdd':
        vtree_filename = '../data/sddFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'.vtree'
        sdd_filename = '../data/sddFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'.sdd'
        lit_edge_file = '../data/sddFiles/'+str(options.mapSize)+'/'+'lit_edge_map_'+str(options.mapSize)+'_zdd.txt'
        save_name = '../data/jsonFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_'
        sdd2jsons(vtree_filename, sdd_filename, lit_edge_file, save_name)
    else:
        json_file = None
        vtree_filename = '../data/sddFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'.vtree'
        psdd_filename = '../data/sddFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'.psdd'
        edge_var_file = None
        save_name = '../data/jsonFiles/'+str(options.mapSize)+'/'+str(options.mapSize)+'_'+str(options.mapType)+'_l'+str(options.numLandmarks)+'_'
        psdd2jsons(json_file, vtree_filename, psdd_filename, edge_var_file, save_name)

if __name__ == '__main__':
    main()









