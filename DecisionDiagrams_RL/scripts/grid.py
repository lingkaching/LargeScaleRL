from graphillion import GraphSet
import itertools

m, n = 5, 5
total_nodes = m*n

def get_lit_edge_dict(universe):
    lit_edge_dict = {}
    i = 1
    for e in universe:
        lit_edge_dict[i] = e
        i += 1
    
    return lit_edge_dict

def save_file(filename, lit_edge_dict):
    f = open(filename, 'w')
    for k,v in lit_edge_dict.items():
        f.write(str(k))
        f.write(" ")
        f.write(str(v))
        f.write("\n")
    f.close()

def get_supergrid_edges(m, n):
    edges = []
    for v in range(1, m*n+1):
        if  v%n != 0:
            edges.append((v,v+1))
        if v <= (m-1)*n: 
            edges.append((v,v+n))
    # super_nodes = [total_nodes+1, total_nodes+2]
    super_edges = [(1,total_nodes+1), (2, total_nodes+2), (3,total_nodes+3), (4,total_nodes+4), (5,total_nodes+5),
    (total_nodes-4,total_nodes+6), (total_nodes-3, total_nodes+7), (total_nodes-2, total_nodes+8), (total_nodes-1, total_nodes+9), (total_nodes, total_nodes+10)]

    edges = edges + super_edges
    
    return edges

edges = get_supergrid_edges(m, n)

GraphSet.set_universe(edges)
universe = GraphSet.universe()

path_cache = {}
for i, j in itertools.combinations(range(1,total_nodes+10+1), 2):
    path_cache[(min(i,j), max(i,j))] = GraphSet.paths(i,j)

paths = GraphSet()
for path in path_cache.values():
    paths = paths.union(path)

filename = "5x5_icaps"
zdd_file = filename + ".zdd"
f = open(zdd_file,'w')
paths.dump(f)
f.close()
print("ZDD SAVED!")

lit_edge_dict = get_lit_edge_dict(universe)
lit_edge_map_file = "lit_edge_map_" + filename + "_zdd.txt"
save_file(lit_edge_map_file, lit_edge_dict)
print("LIT TO EDGE MAP SAVED!")