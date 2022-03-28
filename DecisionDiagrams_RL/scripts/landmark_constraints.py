from pysdd.sdd import Vtree, SddManager
from itertools import combinations
import random
import json
import networkx as nx

random.seed(1)

# Input files
edge_var_file = "../data/sddFiles/10x10/edge_var_map_10x10.txt"
vtree_file = "../data/sddFiles/10x10/10x10.vtree"
## Comment out for smaller maps (e.g. 5x5)
json_file = "../data/sddFiles/10x10/10x10.json"

# Initialize vtree and sdd manager
vtree = Vtree.from_file(vtree_file.encode())
mgr = SddManager.from_vtree(vtree)
var_count = vtree.var_count()
print("var count: ", var_count)


x = [None] + [mgr.literal(i) for i in range(1, var_count+1)]


class JsonMap:
    """ Loads map information from json files """

    def get_lit_edge_map(self, filename):
        lit_edge_map = {}
        with open(filename) as f:
            for line in f:
                x = line.split()
                a, b = x[1].strip('(,)'), x[2].strip('(,)')
                lit_edge_map[int(x[0])] = (int(a), int(b))
        edge_lit_map = {v:k for k,v in lit_edge_map.items()}

        return lit_edge_map, edge_lit_map

    def get_graph_json(self, json_file):
        with open(json_file) as f:
            grid = json.load(f)
        
        grid_edges = grid['edges']
        edges = []
        for e in grid_edges:
            edges.append((e['x'], e['y']))
        
        clusters = grid['clusters']

        g = nx.Graph()
        g.add_edges_from(edges)
        
        return g, clusters

    def get_edge_to_var_dict(self, edge_var_file):
        name_lit_map = {}
        with open(edge_var_file) as f:
            for line in f:
                (k,v) = line.split()
                name_lit_map[int(k)] = int(v)

        return name_lit_map
            
    def get_edge_to_name_dicts(self, json_file):
        edge_name_map = {}
        with open(json_file) as f:
            grid = json.load(f)

        edges = grid['edges']

        for e in edges:
            edge = (e['x'], e['y'])
            edge_name_map[edge] = str(e['name'])

        name_edge_map = {v:k for k,v in edge_name_map.items()}

        return edge_name_map, name_edge_map

    def get_var_to_edge_dict(self, name_lit_map, name_edge_map):
        lit_edge_map = {}
        for name,lit in name_lit_map.items():
            edge = name_edge_map[str(name)]
            lit_edge_map[lit] = edge

        edge_lit_map = {v:k for k,v in lit_edge_map.items()}
        return lit_edge_map, edge_lit_map


def get_graph(m, n):
    """ Get a networkx object for small map sizes (e.g. 5x5).
    Change super_edges according to your map, sources, and destinations. 
    """
    edges = []
    total_nodes = m*n
    for v in range(1, m*n+1):
        if  v%n != 0:
            edges.append((v,v+1))
        if v <= (m-1)*n: 
            edges.append((v,v+n))

    super_edges = [(1,total_nodes+1), (2, total_nodes+2), (3,total_nodes+3), (4,total_nodes+4), (5,total_nodes+5), (total_nodes-4,total_nodes+6), 
                (total_nodes-3, total_nodes+7), (total_nodes-2, total_nodes+8), (total_nodes-1, total_nodes+9), (total_nodes, total_nodes+10)]

    edges = edges + super_edges

    g = nx.Graph(edges)

    return g

# Uncomment for smaller maps
# g = get_graph(5, 5)
# lit_edge_map, edge_lit_map = get_lit_edge_map(edge_var_file)

# Comment out for smaller maps
json_map = JsonMap()
g, clusters = json_map.get_graph_json(json_file)
name_lit_map = json_map.get_edge_to_var_dict(edge_var_file)
edge_name_map, name_edge_map = json_map.get_edge_to_name_dicts(json_file)
lit_edge_map, edge_lit_map = json_map.get_var_to_edge_dict(name_lit_map, name_edge_map)


# Specify landmarks here
n_landmarks = 10
landmarks = [1, 2, 3, 4, 10, 11, 12, 97, 98, 99]

# randomly sample landmarks
# landmarks = []
# landmarks = random.sample(range(1, 26), k=n_landmarks) 

print("n_landmarks: {}, landmarks: {}, len: {} ".format(n_landmarks, landmarks, len(landmarks)))
individual_landmark_lits = []

for l in landmarks:
    temp_edges = list(g.edges(l))
    edges = []
    for e in temp_edges:
        if e[0]>e[1]:
            edges.append((e[1], e[0]))
        else:
            edges.append(e)
    
    lits = [edge_lit_map[e] for e in edges]
    individual_landmark_lits.append(lits)

landmark_lits = []
for i in range(n_landmarks):
    landmark_lits.append(individual_landmark_lits[i])


sdds = []
for _ in range(n_landmarks):
    sdds.append(mgr.false())
final_sdd = mgr.true()
for n in range(n_landmarks):
    incident_lits = landmark_lits[n]
    for e in incident_lits:
        sdds[n] |= x[e]
    
    final_sdd &= sdds[n] 

# for n in range(n_landmarks):
#     incident_edges = landmark_lits[n]
#     for var in combinations(incident_edges, 2):
#         v1, v2 = var[0], var[1]
#         selected_vars = {v1, v2}
#         if len(incident_edges)==2:
#             sdds[n] |= x[v1] & x[v2]
#         elif len(incident_edges)==3:
#             neg_var = list(set(incident_edges) - selected_vars)
#             sdds[n] |= x[v1] & x[v2] & ~x[neg_var]
#         elif len(incident_edges)==4:
#             neg_vars = list(set(incident_edges) - selected_vars)
#             sdds[n] |= x[v1] & x[v2] & ~x[neg_vars[0]] & ~x[neg_vars[1]]

#     mgr.save(("3x3super_min_constr"+str(n)+".sdd").encode(), sdds[n])
#     print("SDD SAVED!")    

# for e1 in l1_edges:
#     for e2 in l2_edges:
#         sdd |= x[e1] & x[e2]

mgr.save(("constraint.sdd").encode(), final_sdd)
print("CONSTRAINT SDD SAVED!")
