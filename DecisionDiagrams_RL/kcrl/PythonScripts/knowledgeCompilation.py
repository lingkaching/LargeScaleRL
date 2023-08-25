# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:47:07 2020

@author: jjling.2018
"""


import networkx as nx
import json
import random
import os

class my_SddNode:
    #FALSE,TRUE,LITERAL,DECOMPOSITION = 0,1,2,3
    def __init__(self, vtree_id, node_id, node_type, literal, elements):
        self.id = node_id
        self.node_type = node_type
        self.literal = literal
        self.elements = elements
        self.vtree_id = vtree_id

class TDSAT:
	def __init__(self, node_id_list_file, node_type_list_file, elements_list_file, literal_list_file, vtree_id_list_file, vtree_ancessors_file, edge_var_file, json_file, lit_edge_file, s, d):

		if os.path.isfile(lit_edge_file):
			self.edge_lit_map, self.lit_edge_map, self.edges = self.generate_mapping_files_zdd(lit_edge_file)
		else:
			self.edge_lit_map, self.lit_edge_map, self.edges = self.generate_mapping_files_json(edge_var_file, json_file)

		self.s_lit = self.edge_lit_map[s]

		self.d_lit = self.edge_lit_map[d]
        
		self.construct_sdd(node_id_list_file, node_type_list_file, elements_list_file, literal_list_file, vtree_id_list_file, vtree_ancessors_file)

		self.evi_edges = [self.d_lit, self.s_lit]

	def reset(self):
		self.evi_edges = [self.d_lit, self.s_lit]
		self.nodes_info = {k: None for k in range(self.num_psdd_node)}

	def construct_sdd(self, node_id_list_file, node_type_list_file, elements_list_file, literal_list_file, vtree_id_list_file, vtree_ancessors_file):

		with open(node_id_list_file) as f:
			node_id_list = json.load(f)
		with open(node_type_list_file) as f:
			node_type_list = json.load(f)
		with open(elements_list_file) as f:
			elements_list = json.load(f)
		with open(literal_list_file) as f:
			literal_list = json.load(f)
		with open(vtree_id_list_file) as f:
			vtree_id_list = json.load(f)
		with open(vtree_ancessors_file) as f:
			vtree_ancessors = json.load(f)

		self.num_psdd_node = len(node_id_list) 
		self.num_vtree_node = max(vtree_id_list)+1
		
		self.vtree_ancessors= {}
		for key in vtree_ancessors.keys():
			self.vtree_ancessors[int(key)] = {i:False for i in range(self.num_vtree_node)}
			for j in vtree_ancessors[key]:
				self.vtree_ancessors[int(key)][j] = True

		my_SddNodes = []
		id_map = {}
		for i in range(len(node_id_list)):
			vtree_id = vtree_id_list[i]
			node_id = node_id_list[i]
			node_type = node_type_list[i]
			node_literal = literal_list[i]
			node_elements = None
			id_map[node_id] = i
			my_SddNodes.append(my_SddNode(vtree_id, node_id, node_type, node_literal, node_elements))

		for i in range(len(node_id_list)):
			if elements_list[i] != None:
				my_SddNodes[i].elements = []
				for p,s in elements_list[i]:
					my_SddNodes[i].elements.append((my_SddNodes[id_map[p]], my_SddNodes[id_map[s]]))
		self.sdd_root = my_SddNodes[-1]
		return

	def generate_mapping_files_json(self, edge_var_file, json_file):
		#reading text file 
		name_lit_map = {}
		with open(edge_var_file) as f:
			for line in f:
				(k,v) = line.split()
				name_lit_map[int(k)] = int(v)

		#reading json file 
		edge_name_map = {}
		with open(json_file) as f:
			grid = json.load(f)
		edges = grid['edges']
		for e in edges:
			edge = (e['x'], e['y'])
			edge_name_map[edge] = int(e['name'])
		name_edge_map = {edge_name_map[k]:k for k in edge_name_map.keys()}

		#construct lit_edge_map and edge_lit_map
		lit_edge_map = {}
		for name in name_lit_map.keys():
			lit = name_lit_map[name]
			edge = name_edge_map[name]
			lit_edge_map[lit] = edge
		edge_lit_map = {lit_edge_map[k]:k for k in lit_edge_map.keys()}

		#generate neighbouring edges
		edges = {}
		all_edges = list(edge_lit_map.keys())
		for i in range(0, len(all_edges)):
			edge = all_edges[i]
			edges[edge_lit_map[edge]] = []
			for j in range(0, i):
				if edge[0] in all_edges[j] or edge[1] in all_edges[j]:
					edges[edge_lit_map[edge]].append(edge_lit_map[all_edges[j]])
			for j in range(i+1, len(all_edges)):
				if edge[0] in all_edges[j] or edge[1] in all_edges[j]:
					edges[edge_lit_map[edge]].append(edge_lit_map[all_edges[j]])
		return edge_lit_map, lit_edge_map, edges

	def generate_mapping_files_zdd(self, lit_edge_file):
		lit_edge_map = {}
		with open(lit_edge_file) as f:
			for line in f:
				x = line.split()
				a, b = x[1].strip('(,)'), x[2].strip('(,)')
				lit_edge_map[int(x[0])] = (int(a), int(b))
		edge_lit_map = {lit_edge_map[k]:k for k in lit_edge_map.keys()}

		#generate neighbouring edges
		edges = {}
		all_edges = list(edge_lit_map.keys())
		for i in range(0, len(all_edges)):
			edge = all_edges[i]
			edges[edge_lit_map[edge]] = []
			for j in range(0, i):
				if edge[0] in all_edges[j] or edge[1] in all_edges[j]:
					edges[edge_lit_map[edge]].append(edge_lit_map[all_edges[j]])
			for j in range(i+1, len(all_edges)):
				if edge[0] in all_edges[j] or edge[1] in all_edges[j]:
					edges[edge_lit_map[edge]].append(edge_lit_map[all_edges[j]])
		
		return edge_lit_map, lit_edge_map, edges

	def compatible(self, literal, evi):
		if literal not in evi and literal*-1 not in evi:
			return True
		if literal in evi:
			return True
		else:
			return False
	
	def top_down_search(self, root, evi, nodes_info, visited_nodes, vtree_ancessors):
		if root.node_type == 3:
			flag = False
			for p,s in root.elements:
				if s.node_type == 0: continue
				
				if not vtree_ancessors[p.vtree_id] and nodes_info[p.id] == True:
					visited_nodes[p.id] = True
					# if nodes_info[p.id]:					
					if not vtree_ancessors[s.vtree_id] and nodes_info[s.id] != None:
						flag = nodes_info[s.id]
						visited_nodes[s.id] = flag					
					elif visited_nodes[s.id] != None:
						flag = visited_nodes[s.id]
					else:
						flag, visited_nodes = self.top_down_search(s, evi, nodes_info, visited_nodes, vtree_ancessors)
						visited_nodes[s.id] = flag				
					if flag:
						break
			
				if  visited_nodes[p.id] == True:
					# print(s.id)
					# print(s.vtree_id)
					if not vtree_ancessors[s.vtree_id] and nodes_info[s.id] != None:
						flag = nodes_info[s.id]
						visited_nodes[s.id] = flag
					if visited_nodes[s.id] != None:
						flag = visited_nodes[s.id]
					else:
						flag, visited_nodes = self.top_down_search(s, evi, nodes_info, visited_nodes, vtree_ancessors)
						visited_nodes[s.id] = flag
					if flag:
						break

				elif  visited_nodes[p.id] == None:
					if p.node_type == 2:
						if self.compatible(p.literal, evi):
							visited_nodes[p.id] = True
							if not vtree_ancessors[s.vtree_id] and nodes_info[s.id] != None:
								flag = nodes_info[s.id]
								visited_nodes[s.id] = flag
							if visited_nodes[s.id] != None:
								flag = visited_nodes[s.id]
							else:
								flag, visited_nodes = self.top_down_search(s, evi, nodes_info, visited_nodes, vtree_ancessors)
								visited_nodes[s.id] = flag
							if flag:
								break
						else:
							visited_nodes[p.id] = False       					
					# p is a decompostion node
					else:
						flag_p,  visited_nodes = self.top_down_search(p, evi, nodes_info, visited_nodes, vtree_ancessors)
						if flag_p:
							visited_nodes[p.id] = True
							if not vtree_ancessors[s.vtree_id] and nodes_info[s.id] != None:
								flag = nodes_info[s.id]
								visited_nodes[s.id] = flag
							if visited_nodes[s.id] != None:
								flag = visited_nodes[s.id]
							else:
								flag, visited_nodes = self.top_down_search(s, evi, nodes_info, visited_nodes, vtree_ancessors)
								visited_nodes[s.id] = flag
							if flag:
								break
						else:
							visited_nodes[p.id] = False   
			return flag, visited_nodes

		if root.node_type == 1: return True, visited_nodes

		if not vtree_ancessors[root.vtree_id] and nodes_info[root.id] != None: 
			flag = nodes_info[root.id]
			visited_nodes[root.id] = flag
			return flag, visited_nodes

		if visited_nodes[root.id] != None:
			return visited_nodes[root.id], visited_nodes
		else:
			if self.compatible(root.literal, evi):
				visited_nodes[root.id] = True
				return True, visited_nodes
			else:
				visited_nodes[root.id] = False
				return False, visited_nodes

	def get_valid_literal(self):
		successors = []
		candidate_edge = [edge for edge in self.edges[self.evi_edges[-1]] if edge not in self.evi_edges] 
		visited_nodes_dict = {}
		for edge in candidate_edge:
			temp, visited_nodes = self.top_down_search(self.sdd_root, self.evi_edges+[edge], self.nodes_info, {k: None for k in range(self.num_psdd_node)}, self.vtree_ancessors[edge])
			if temp:
				successors.append(edge)
				visited_nodes_dict[edge] = visited_nodes
		return successors, visited_nodes_dict

