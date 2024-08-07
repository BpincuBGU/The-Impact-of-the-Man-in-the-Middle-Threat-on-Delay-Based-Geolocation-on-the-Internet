import numpy as np
import math
import tqdm
import warnings
import time
import heapq
import os
import sys
import networkx as nx
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import multiprocessing
import ast

 
warnings.simplefilter('error')
PROBS_NUM = 5
CR_NUM = 5
EUROPE = ["FR","DE","UA","ES","SE","FI","NO","PL","IT","GB","RO"] # Biggest countries in Europe
# Countries topology to load, if empty load all countries
COUNTRIES = ['FR'] 
# Can specify cities to load, if empty load all cities
CITIES = []
COOPERATIVE = True
 
NODES_PATH = "./topologyNew.txt"
LINKS_PATH = "./linksFinal.txt"
 
Impact = {}
Shortest_Path_Counter = 0
Name_To_Index = {}
Index_To_Name = {}
Shortest_Distances = 0
DIPs = {}
Tsigma = {}
Vertex_To_Index = {}
Nodes_To_Pos = []

states_to_names = {}

# round the edge weights, this help avoid floating point errors and create a more realistic simulation with more possible paths
roundPow = 2
roundCoef = 10**roundPow


'''
    Read the topology from the files NODES_PATH and LINKS_PATH
    Returns a networkx graph with the topology
'''
def read_topology_networkx():
    # initialize all the dictionaries
    print(COUNTRIES)
    global Impact
    global Shortest_Path_Counter
    global Name_To_Index
    global Index_To_Name
    global Shortest_Distances
    global DIPs
    global Tsigma
    global Vertex_To_Index
    global Nodes_To_Pos
    Impact = {}
    Shortest_Path_Counter = {}
    Name_To_Index = {}
    Index_To_Name = {}
    Shortest_Distances = {}
    DIPs = {}
    Tsigma = {}
    Vertex_To_Index = {}
    counter = 0
    g = nx.Graph()
    # Read node data from NODES_PATH
    with open(NODES_PATH, 'r', encoding='utf8') as f:
        nodes = f.readlines()
    for node_info in nodes:
        name, cont , country, area , city, longitude, latitude, _,_,_ = node_info.split("\t")
        name = name.split(" ")[1][:-1]
        if longitude[-1] == "\n":
            latitude = latitude[:-1]
        if (country in COUNTRIES or not COUNTRIES) and (city in CITIES or not CITIES):
            g.add_node(name, longitude=float(longitude), latitude=float(latitude), pos=(float(latitude), float(longitude)))
            counter += 1
            if(COUNTRIES == ["US"]):
                if( area in states_to_names):
                    states_to_names[area].append(name)
                else:
                    states_to_names[area] = [name]
    # Read links data from LINKS_PATH
    Nodes_To_Pos = nx.get_node_attributes(g, "pos")
    with open(LINKS_PATH, 'r') as f:
        links = f.readlines()
    for link in links:
        if(link[-1] == "\n"):
            link = link[:-1]
        for node in link.split(" "):
            if node in g.nodes:
                for node2 in link.split(" "):
                    if node2 in g.nodes:
                        if not node2 == node:
                            g.add_edge(node, node2, weight = compute_distance(g, node, node2))
    # remove isolated nodes from the graph and all the dictionaries
    for node in list(g.nodes):
        if g.degree(node) == 0:
            g.remove_node(node)
    # index all the nodes in the graph
    for i, node in enumerate(g.nodes()):
        Name_To_Index[node] = i
        Index_To_Name[i] = node
    # for edge in list(g.edges):
    #     if g.edges[edge]["weight"] == 0:
    #         # set the edge weight to 0.1
    #         g.edges[edge]["weight"] = 0.1
    # # print the weight of the edge with min
    print("Min edge weight: ", min([g.edges[edge]["weight"] for edge in g.edges]))
    print("Max edge weight: ", max([g.edges[edge]["weight"] for edge in g.edges]))
    # print the edge with min weight
    print("Min edge: ", min(g.edges, key=lambda x: g.edges[x]["weight"]))
    print("Number of nodes: ", g.number_of_nodes())
    print("Number of edges: ", g.number_of_edges())
    print("Size of graph in GB: ", g.number_of_edges()*100/1024/1024/1024)
    return g
 
'''
    Compute the distance between two nodes using the Haversine formula
    Returns the distance between the two nodes in kilometers
'''
def compute_distance(g, node1, node2):
    # Radius of the Earth in kilometers
    R = 6371.0
    coord1 = Nodes_To_Pos[node1]
    coord2 = Nodes_To_Pos[node2]
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
 
    # Calculate the differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
 
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
 
    # Calculate the distance
    distance = R * c
 
    return int(max(round(distance,roundPow),1/roundCoef)*roundCoef) # originally returned only distance
 
'''
    Returns the shortest distance between two nodes
'''
def get_shortest_distance(a,b):
    global Shortest_Distances
    return Shortest_Distances[Name_To_Index[a],Name_To_Index[b]]
 
'''
    Returns the number of shortest paths between two nodes
'''
def get_shortest_path_counter(a,b):
    global Shortest_Path_Counter
    return Shortest_Path_Counter[Name_To_Index[a],Name_To_Index[b]]
 
'''
    Set the shortest distance between two nodes
'''
def set_shortest_distance(a, b, dist):
    global Shortest_Distances
    Shortest_Distances[Name_To_Index[a], Name_To_Index[b]] = dist
 
'''
    Set the number of shortest paths between two nodes
'''
def set_shortest_path_counter(a, b, count):
    global Shortest_Path_Counter
    Shortest_Path_Counter[Name_To_Index[a], Name_To_Index[b]] = count
 
'''
    Modified Dijkstra algorithm to calculate the shortest distances and path counters
    Receives a graph, a list of start nodes, local dictionaries and a dictionary to map node names to indices
    Returns nothing, but updates the local dictionaries with the results for each start node
'''
def modified_dijkstra(g, start_nodes, local_dictinaries, Name_To_Index):
    for start_node in start_nodes:
        # Initialize distances and path counters
        distances = np.full(len(g.nodes()),float("inf"), dtype=object)
        path_counters = np.zeros(len(g.nodes()), dtype=object)
        distances[Name_To_Index[start_node]] = 0
        path_counters[Name_To_Index[start_node]] = 1
        queue = []
        # Priority queue to store nodes to be visited
        heapq.heappush(queue, (0, start_node))
 
        while queue:
            current_distance, current_node = heapq.heappop(queue)
 
            # Ignore nodes that have already been visited with a shorter distance
            if current_distance > distances[Name_To_Index[current_node]]:
                continue
 
            # Explore neighbors of the current node
            for neighbor in g.neighbors(current_node):
                weight = g[current_node][neighbor]['weight']
                distance = current_distance + weight
 
                # Update distance and path counter if a shorter path is found
                if distance < distances[Name_To_Index[neighbor]]:
                    distances[Name_To_Index[neighbor]] = distance
                    path_counters[Name_To_Index[neighbor]] = 0
                    # remove the neighbor from the queue if it is already there
                    pushed = False
                    for i in range(len(queue)):
                        if queue[i][1] == neighbor:
                            queue[i] = (distance, neighbor)
                            heapq.heapify(queue)
                            pushed = True
                            break
                    if not pushed:
                        heapq.heappush(queue, (distance, neighbor))
 
                # If there is another shortest path to the neighbor, increment the path counter
                if distance == distances[Name_To_Index[neighbor]]:
                    path_counters[Name_To_Index[neighbor]] += path_counters[Name_To_Index[current_node]]
 
        # Update the global dictionaries with the results for the start node
        for node in g.nodes():
            local_dictinaries[Name_To_Index[start_node], Name_To_Index[node]] = (distances[Name_To_Index[node]], path_counters[Name_To_Index[node]])
 
'''
    Worker function to calculate the shortest distances and path counters for a chunk of start nodes
    Receives a tuple with the graph, the start nodes, and a dictionary to map node names to indices
    Returns a tuple with the start nodes and the local dictionaries
'''
def compute_shortest_distances_worker(args):
    g, start_nodes, Name_To_Index = args
    local_dictionaries = np.zeros((g.number_of_nodes(), g.number_of_nodes()), dtype=object)
    # now use the modified dijkstra to calculate the shortest distances and path counters
    for source in tqdm.tqdm(start_nodes, total=len(start_nodes)): 
        modified_dijkstra(g, [source], local_dictionaries, Name_To_Index)
    return start_nodes, local_dictionaries
 

''' 
    Pre calculate the shortest distances and path counters for all pairs of nodes in the graph
    Receives a graph and the number of processes to use
    Returns nothing, but updates the global dictionaries with the results
'''
def pre_calculations(g, num_processes=os.cpu_count()):
    global Shortest_Distances
    global Shortest_Path_Counter
    # if exist files with the pre calculations, load them
    if os.path.exists("Shortest_Distances"+str(COUNTRIES)+".npy") and os.path.exists("Shortest_Path_Counter"+str(COUNTRIES)+".npy"):
        print("Reading pre calculations from file")
        Shortest_Distances = np.load("Shortest_Distances"+str(COUNTRIES)+".npy", allow_pickle=True, mmap_mode='c')
        Shortest_Path_Counter = np.load("Shortest_Path_Counter"+str(COUNTRIES)+".npy", allow_pickle=True, mmap_mode='c')
        return
    Shortest_Distances = np.full((g.number_of_nodes(), g.number_of_nodes()), -1, dtype=np.float64)
    Shortest_Path_Counter = np.zeros((g.number_of_nodes(), g.number_of_nodes()), dtype=np.ulonglong)
    # Divide start nodes into chunks for each process
    node_chunks = np.array_split(list(g.nodes()), num_processes-1)
    print(num_processes)
    # Create and start worker pool
    pool = multiprocessing.Pool(processes=num_processes-1)
    args = [(g, chunk, Name_To_Index) for chunk in node_chunks]
    # Map worker function to chunks of start nodes
    results = pool.map(compute_shortest_distances_worker, args)
    pool.close()
    # Wait for the worker processes to terminate
    pool.join()
    print("Computing shortest distances and path counters finished")
    # Aggregate results from workers
    for start_nodes, local_dictionary in results:
        for start_node in start_nodes:
            for node in g.nodes():
                set_shortest_distance(start_node, node, local_dictionary[Name_To_Index[start_node], Name_To_Index[node]][0])
                set_shortest_path_counter(start_node, node, local_dictionary[Name_To_Index[start_node], Name_To_Index[node]][1])
    np.save("Shortest_Distances"+str(COUNTRIES)+".npy", Shortest_Distances, allow_pickle=True)
    np.save("Shortest_Path_Counter"+str(COUNTRIES)+".npy", Shortest_Path_Counter, allow_pickle=True)
    print("Max distance: ", np.max(Shortest_Distances))
    print("Min distance: ", np.min(Shortest_Distances))
    print("Max path counter: ", np.max(Shortest_Path_Counter))
    print("Min path counter: ", np.min(Shortest_Path_Counter))
    print("Finished Pre Calculations")

'''
    Pre calculate the shortest distances and path counters for all pairs of nodes in the graph using a single CPU
    Receives a graph
    Returns nothing, but updates the global dictionaries with the results
''' 
def pre_calculations_single_cpu(g):
    start = time.time()
    q = []
    global Shortest_Distances
    global Shortest_Path_Counter
    # set infinite distance for all pairs
    print("Initializing")
    Shortest_Path_Counter = np.zeros((g.number_of_nodes(), g.number_of_nodes()), dtype=np.ulonglong)
    Shortest_Distances =np.full((g.number_of_nodes(), g.number_of_nodes()),float("inf"), dtype=np.float64)
    print(sys.getsizeof(Shortest_Path_Counter))
    print(sys.getsizeof(Shortest_Distances))
    print("Starting Pre Calculations")
    for s in tqdm.tqdm(g.nodes(), total=g.number_of_nodes()):
        set_shortest_distance(s,s,0)
        set_shortest_path_counter(s,s,1)
        maxIndex = max(Name_To_Index.values())+1
        dist = np.full(maxIndex, float("inf"), dtype=object)
        sigma = np.zeros(maxIndex, dtype=object)
        dist[Name_To_Index[s]] = 0
        sigma[Name_To_Index[s]] = 1
        heapq.heappush(q, (0, s))
 
        while q:
            # extract v from q with min dist[v]
            minDist, v = heapq.heappop(q)
            if minDist > dist[Name_To_Index[v]]:
                continue
            # S.append(v)
            for w in nx.neighbors(g, v):
                # path discovery
                if dist[Name_To_Index[w]] > dist[Name_To_Index[v]] + g.edges[(v, w)]["weight"]:
                    dist[Name_To_Index[w]] = dist[Name_To_Index[v]] + g.edges[(v, w)]["weight"]
                    sigma[Name_To_Index[w]] = 0
                    set_shortest_distance(s,w,dist[Name_To_Index[w]])
                    # heapq.heappush(q, (dist[Vertex_To_Index[w]], w))
                    pushed = False
                    for i in range(len(q)):
                        if q[i][1] == w:
                            q[i] = (dist[Name_To_Index[w]], w)
                            heapq.heapify(q)
                            pushed = True
                            break
                    if not pushed:
                        heapq.heappush(q, (dist[Name_To_Index[w]], w))
                # path counting
                if dist[Name_To_Index[w]] == dist[Name_To_Index[v]] + g.edges[(v, w)]["weight"]:
                    sigma[Name_To_Index[w]] += sigma[Name_To_Index[v]]
                    set_shortest_path_counter(s,w,sigma[Name_To_Index[w]])
    end = time.time()
    print("Time: ", end - start)
    print("Max distance: ", np.max(Shortest_Distances))
    print("Min distance: ", np.min(Shortest_Distances))
    print("Max path counter: ", np.max(Shortest_Path_Counter))
    print("Min path counter: ", np.min(Shortest_Path_Counter))
    print("Finished Pre Calculations")

'''
    Calculate the DIPs for a given set of compromised routers, probing routers and a target
    Receives a graph, a set of compromised routers, a set of probing routers and a target
    Returns nothing, but updates the global DIPs dictionary with the results
'''
def calculate_DIPS(g, CR, PP, PH):
    global DIPs
    CR = set(CR)
    PP = list(PP)
    PH = list(PH)
    DIPs = {}
    for p in PP:
        if p in CR:
            for h in PH:
                DIPs.update({(p,h) : {0 : 1}})
            continue
        else:
            CR = sorted(CR, key=lambda x: get_shortest_distance(p,x))
            PSCR = []
            for cr in CR:
                Tsigma[(p,cr)] = get_shortest_path_counter(p,cr)
                for c in PSCR:
                    if cr != c and get_shortest_distance(p,cr) == get_shortest_distance(p,c) + get_shortest_distance(c,cr):
                        Tsigma[(p,cr)] = Tsigma[(p,cr)] - Tsigma[(p,c)]*get_shortest_path_counter(c,cr)
                if Tsigma[(p,cr)] > 0:
                    PSCR.append(cr)
            for h in PH:
                SCR = []
                for cr in PSCR:
                    if get_shortest_distance(p,h) == get_shortest_distance(p,cr) + get_shortest_distance(cr,h):
                        SCR.append(cr)
                        CompromisedRoutes = 0
                        for c in SCR:
                            CompromisedRoutes += Tsigma[(p,c)] * get_shortest_path_counter(c,h)
                        if get_shortest_path_counter(p,h) > 0:
                            if (p,h) in DIPs:
                                DIPs[(p,h)].update({get_shortest_distance(p,cr) : CompromisedRoutes/get_shortest_path_counter(p,h)})
                            else:
                                DIPs.update({(p,h): {get_shortest_distance(p,cr) : CompromisedRoutes/get_shortest_path_counter(p,h)}})
            # print("SCR: ", SCR)
            # print("PSCR: ", PSCR)
        for h in PH:
            if (p,h) not in DIPs:
                DIPs.update({(p,h) : {0 : 0}})
 
'''
    Gets the accurate DiP for a given probing router, compromised router and target from the DIPs dictionary
    Receives a graph, a probing router, a compromised router and a target
    Returns the DiP value
'''
def DiP(g, p, h, x):
    if COOPERATIVE and get_shortest_distance(p,x) >= get_shortest_distance(p,h):
        return 1
    dist = get_shortest_distance(p,x)
    probs = 1
    if dist == -1:
        return 0
    closest = None
    for key in DIPs[(p,h)]:
        if key <= dist:
            closest = key
    if closest == None:
        return 0
    else:
        probs *= DIPs[(p,h)][closest]
    return probs

'''
    Gets the accurate DiP for a given group of probing routers, compromised routers and a target from the DIPs dictionary
    Receives a graph, a group of probing routers, a group of compromised routers and a target
    Returns the DiP value
'''
def DiPGroup(g, P, h, x):
    res = 1
    for p in P:
        res *= DiP(g, p,h,x)
    return res

'''
    Gets the average impact of a group of probing routers on a target
    Receives a graph, a group of probing routers and a target
    Returns the average impact
'''
def AverageImpact(g, P, PH):
    impact = 0
    for x in g.nodes():
            impact += compute_distance(g, x, PH) * DiPGroup(g, P, PH, x)
    return (impact/g.number_of_nodes())/roundCoef
 
'''
    Gets the maximum impact of a group of probing routers on a target
    Receives a graph, a group of probing routers and a target
    Returns the maximum impact
'''
def MaxImpact(g, P, PH):
    return max([compute_distance(g, x, PH) * DiPGroup(g, P, PH, x) for x in g.nodes()])/roundCoef
 
'''
    Recieves a graph, a group of probing routers, a group of compromised routers and a target
    Draws the graph with the nodes colored according to their probability of being compromised
    and a graph of the network with the nodes colored according to their role
'''
def save_data(g, CR, PP, PH, index):
    H = PH[0]
    chosen_probes = PP
    impact = AverageImpact(g, PP, H)
    print("Average Impact: ", impact)
    pos = nx.get_node_attributes(g, "pos")
    color = []
    nodes_size = []
    for v in g.nodes():
        if v in PH:
            nodes_size.append(10)
            color.append("green")
        elif v in CR:
            color.append("red")
            nodes_size.append(10)
        elif v in PP:
            color.append("blue")
            nodes_size.append(10)
        else:
            color.append("grey")
            nodes_size.append(0.7)
    if(os.path.exists("graphs/"+str(int(impact))) == False):
        os.mkdir("graphs/"+str(int(impact)))
    path = "graphs/"+str(int(impact))
    plt.figure(num=1, clear=True)
    colorEdge = []
    for e in g.edges(): # if e is on a shortest path between any p in PP to H, color it red
        added = False
        for p in PP:
            if get_shortest_distance(p,e[0]) + g.edges[e]["weight"] + get_shortest_distance(e[1],H) == get_shortest_distance(p,H):
                colorEdge.append("red")
                added = True
                break
            elif get_shortest_distance(p,e[1]) + g.edges[e]["weight"] + get_shortest_distance(e[0],H) == get_shortest_distance(p,H):
                colorEdge.append("red")
                added = True
                break
        if not added:
            colorEdge.append("black")
    nx.draw_networkx_nodes(g, pos=pos, node_size=nodes_size, node_color=color, node_shape=".", linewidths=0)
    nx.draw_networkx_edges(g, pos=pos, edge_color=colorEdge, width=0.01, alpha=0.05)
    plt.box(False)
    plt.savefig(path+"/graph" + str(index) + ".png", dpi = 2000)
    probs = {}
    for v in g.nodes():
        probs[v] = DiPGroup(g, PP, H, v)
    # give each node color according to its probability
    color = []          
    for v in g.nodes():
        if probs[v] == 0:
            color.append("red")
        elif probs[v] < 0.5:
            color.append("yellow")
        elif probs[v] < 1:
            color.append([0,0.5,0,1])
        else:
            color.append([0,1,0,1])
    plt.figure()
    nx.draw_networkx_nodes(g, pos=pos, node_size=0.7, node_color=color, node_shape=".", linewidths=0)
    nx.draw_networkx_edges(g, pos=pos, edge_color=colorEdge, width=0.01, alpha=0.05)
    plt.box(False)
    plt.savefig(path+"/graph" + str(index) + "probs.png", dpi=2000)
    plt.close()
    # save the dips to a file
    with open(path+"/dips" + str(index) + ".txt", "w") as f:
        f.write("Average Impact: " + str(impact) + "\n")
        f.write("Chosen Probes: " + str([x for x in chosen_probes]) + "\n")
        f.write("Compromised Router: " + str([x for x in CR]) + "\n")
        f.write("Target: " + str(H) + "\n")
        for key in DIPs:
            f.write(str(key) + " : " + str(DIPs[key]) + "\n")
 
'''
Recieves a graph, a group of probing routers, a group of compromised routers, a target and a number of requested compromised routers
Returns a list of the n best compromised routers in this scenario
'''
def get_n_best_crs_specific(g,PP,PH,n):
    bestCrs = []
    for i in range(n):
        maxImpact = 0
        bestCr = None
        for v in g.nodes():
            if not v is PH[0] and g.degree(v) > 1:
                bestCrs.append(v)
                calculate_DIPS(g,bestCrs,PP,PH)
                impact = AverageImpact(g,PP,PH[0])
                if impact > maxImpact:
                    maxImpact = impact
                    bestCr = v
                bestCrs.remove(v)
        bestCrs.append(bestCr)
    return bestCrs
 
'''
    A greedy algorithm to find the n best compromised routers across multiple scenarios
    Receives a graph and the number of requested compromised routers
    Returns a list of the n best compromised routers
'''
def get_n_best_cr(g, n):
    crCounter = {}
    for i in tqdm.tqdm(range(1000)):
        PP = np.random.choice(list(g.nodes()), PROBS_NUM, replace=False)
        PH = list(np.random.choice(list(g.nodes()), 1, replace=False))
        crsFound = get_n_best_crs_specific(g,PP,PH,n)
        for cr in crsFound:
            if cr in crCounter:
                crCounter[cr] += 1
            else:
                crCounter[cr] = 0
    return sorted(crCounter, key=lambda x: crCounter[x], reverse=True)[:n]
 
def get_crs_from_file():
    with open("bestCRS.txt", "r") as f:
        lines = f.read().split("\n")
    for line in lines:
        country, crs = line.split("-")
        if country in COUNTRIES:
            return ast.literal_eval(crs)
    return None

def get_n_best_cr_v3(g, n):
    found_CR = []
    # check if the best CRs are already calculated
    crs = get_crs_from_file()
    if len(crs) >= n:
        return crs[:n]
    elif crs is not None:
        found_CR = crs
        n = n - len(found_CR)
 
    # core_count = int(os.environ['SLURM_CPUS_ON_NODE'])
    core_count = 4
    print(core_count)
    iterations = 100
    iterations_per_process = iterations // core_count
    print("Starting to find CRs")
    # Create a pool of processes
    print(os.cpu_count())
    pool = multiprocessing.Pool(processes=core_count)
    for j in range(n):
        start = time.time()
        print("Finding CR number ", j+1)
        results = pool.starmap(worker_cr_v3, [(g, iterations_per_process, Name_To_Index, Nodes_To_Pos, found_CR.copy(), COUNTRIES)] * core_count)
 
        # Aggregate results
        total_cr_counter = np.sum(results, axis=0)
        best_cr_index = np.argmax(total_cr_counter)
        found_CR.append(Index_To_Name[best_cr_index])
        print(found_CR)
        end = time.time()
        print("Time it took to find CR number ", j+1, ": ", end - start)
 
    # Close the pool of processes
    pool.close()
    pool.join()
    with open("bestCRS.txt", "a") as f:
        f.write("\n" + str(COUNTRIES[0]) + "-" + str(found_CR))
 
    return found_CR
 
def worker_cr_v3(g, iterations_per_process, Name_To_Index_Local, Nodes_To_Pos_Local, found_CR, countries):
    cr_counter = np.zeros(g.number_of_nodes(), dtype=np.int32)
    global Name_To_Index
    global Shortest_Distances
    global Shortest_Path_Counter
    global Nodes_To_Pos
    global COUNTRIES
    COUNTRIES = countries
    Name_To_Index = Name_To_Index_Local 
    Shortest_Distances = np.load("Shortest_Distances"+str(COUNTRIES)+".npy", allow_pickle=True, mmap_mode='c')
    Shortest_Path_Counter = np.load("Shortest_Path_Counter"+str(COUNTRIES)+".npy", allow_pickle=True, mmap_mode='c')
    Nodes_To_Pos = Nodes_To_Pos_Local
    for _ in tqdm.tqdm(range(iterations_per_process)):
        PP = np.random.choice(list(g.nodes()), 5, replace=False)
        PH = np.random.choice(list(g.nodes()), 1, replace=False)
        bestCR = 0
        maxImpact = -math.inf
        for v in g.nodes():
            if g.degree(v) > 1 and v not in found_CR:
                found_CR.append(v)
                calculate_DIPS(g, found_CR, PP, PH)
                impact = AverageImpact(g, PP, PH[0])
                if impact > maxImpact:
                    maxImpact = impact
                    bestCR = v
                found_CR.pop()
        cr_counter[Name_To_Index[bestCR]] += 1
    return cr_counter
'''
    Recieves a list of countries
    Calculates the probability of disguise for each distance after choosing the best CRs over the countries
    Draws random examples of manipulation using the CRs that were found
'''
def check_crs_for_topology(countries = EUROPE):
    global COUNTRIES
    # countries = ["US"]
    COUNTRIES = countries
    g = read_topology_networkx()
    pre_calculations(g)
    crs = get_n_best_cr_v3(g,CR_NUM)
    _,_,_,_, probs = check_crs_for_country(g,crs)
    for i in range(10):
        PP = np.random.choice(list(g.nodes()), PROBS_NUM, replace=False)
        PH = np.random.choice(list(g.nodes()), 1, replace=False)
        calculate_DIPS(g,crs,PP,PH)
        save_data(g,crs,PP,PH,i)
        # calculate_DIPS(g,[],PP,PH)
        # save_data(g,crs,PP,PH,i*10)
    return probs
 
'''
    Recieves a list of countries
    Calculates the probability of disguise for each distance after choosing the best CRs for each country, then averaging the results
'''
def check_crs_for_each_country(countries = EUROPE):
    global COUNTRIES
    distancePerProb = {}
    for country in countries:
        COUNTRIES = [country]
        g = read_topology_networkx()
        pre_calculations(g)
        crs = get_n_best_cr_v3(g,5)
        _,_,_,_, probs = check_crs_for_country(g,crs)
        for distance in probs.keys():
            if distance in distancePerProb.keys():
                distancePerProb[distance] += probs[distance]
            else:
                distancePerProb[distance] = probs[distance]
    for distance in distancePerProb.keys():
        distancePerProb[distance] /= len(countries)
    return distancePerProb
 
'''
    Recieves a graph and a list of compromised routers
    Calculates the probability of disguise for each distance given the compromised routers
'''
def check_crs_for_country(g,crs):
    avgImpacts = []
    maxImpacts = []
    iterationCount = 100
    # dictionaries with keys 0 to 0.9 with jumps of 0.1 with values of 0
    DIP_counter = {round(i*0.1,1):0 for i in range(0,11)}
    maxDistances = []
    totalProbabilityPerDistance = {}
    for iteration in tqdm.tqdm(range(iterationCount)):
        distanceToDIP = [[] for i in range(1000)]
        PP = np.random.choice(list(g.nodes()), 5, replace=False)
        PH = np.random.choice(list(g.nodes()), 1, replace=False)
        calculate_DIPS(g,crs,PP,PH)
        maxDistances.append(max([compute_distance(g, x, PH[0])/roundCoef for x in g.nodes()]))
        avgImpacts.append(AverageImpact(g,PP,PH[0]))
        maxImpacts.append(MaxImpact(g,PP,PH[0]))
        for v in g.nodes():
            dip = DiPGroup(g,PP,PH[0],v)
            distance = int((compute_distance(g, v, PH[0])/roundCoef)//100)
            distanceToDIP[distance].append(dip)
            minDip = int(round(dip,1)*10)
            for i in range(0,minDip+1):
                DIP_counter[round(i/10,1)] += 1
        # run on distanceToDIP, from the highest distance to the lowest
        # remove every empty list from the end of distanceToDIP until the first non empty list
        while len(distanceToDIP[-1]) == 0:
            distanceToDIP.pop()
 
        notHappeningProb = 1
        # create a for loop the runs from the end of distanceToDIP to the start
        for key in range(len(distanceToDIP)-1,-1,-1):
            probability = 1 - np.prod(np.array([1 - x for x in distanceToDIP[key]]))
 
            distanceToDIP[key] = probability * notHappeningProb
            if key in totalProbabilityPerDistance:
                totalProbabilityPerDistance[key] += distanceToDIP[key]
            else:
                totalProbabilityPerDistance[key] = distanceToDIP[key]
            notHappeningProb *= 1-probability
    for key in totalProbabilityPerDistance:
        totalProbabilityPerDistance[key] /= iterationCount
 
    totalProbabilityPerDistance = {k*100: round(v*100,1) for k, v in totalProbabilityPerDistance.items() if round(v*100,1) != 0}
 
    totalDips = sum(DIP_counter.values())
    for i in range(0):
        PP = np.random.choice(list(g.nodes()), 5, replace=False)
        PH = np.random.choice(list(g.nodes()), 1, replace=False)
        calculate_DIPS(g,crs,PP,PH)
        save_data(g,crs,PP,PH,i)
        PP = np.random.choice(list(g.nodes()), 5, replace=False)
        PH = np.random.choice(list(g.nodes()), 1, replace=False)
        calculate_DIPS(g,[],PP,PH)
        save_data(g,crs,PP,PH,i*10)
    return avgImpacts,maxImpacts, DIP_counter, maxDistances, totalProbabilityPerDistance
 
'''
    Validates that adding more compromised routers increases the impact on the target
    Plots the average impact of adding more compromised routers
'''
def validate_cr_addition():
    impacts = {}
    # create a list of all countries in europe
    countries = ["PH","PL", "GB", "AL", "AD", "AM", "AT", "AZ", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FO", "FI", "FR", "GE", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI", "LT", "LU", "MK", "MT", "MD", "NL", "NO", "PL", "PT", "RO", "RU", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "RS", "ME"]
    counter = 0
    for country in countries:
        global COUNTRIES
        COUNTRIES = [country]
        g = read_topology_networkx()
        if not 20 < g.number_of_nodes() < 2500:
            continue
        counter += 1
        pre_calculations(g)
        # 50 times, choose 5 random probes and 1 random target
        for i in range(50):
            PP = np.random.choice(list(g.nodes()), 5, replace=False)
            PH = np.random.choice(list(g.nodes()), 1, replace=False)
            # check the impact of 1 random cr 20 times, then 2 random crs 20 times, etc. until 10 random crs
            for j in range(1,11):
                for z in range(20):
                    CR = np.random.choice(list(g.nodes()), j, replace=False)
                    calculate_DIPS(g, CR, PP, PH)
                    impact = AverageImpact(g, PP, PH[0])
                    if j in impacts:
                        impacts[j].append(impact)
                    else:
                        impacts[j] = [impact]
    # plot the average impact of each number of crs
    x = []
    y = []
    for key in impacts:
        x.append(key)
        y.append(sum(impacts[key])/len(impacts[key]))
    plt.plot(x,y)
    plt.xlabel("Number of CRs")
    plt.ylabel("Average Impact")
    plt.savefig("crs_vs_impact.png", dpi=1000)
    plt.close()
    print("Finished ",counter," countries")

'''
    Validates that adding more probing routers increases the impact on the target
    Plots the average impact of adding more probing routers
'''
def validate_probe_addition():
    impacts = {}
    # create a list of all countries in europe
    countries = ["PH","PL", "GB", "AL", "AD", "AM", "AT", "AZ", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FO", "FI", "FR", "GE", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI", "LT", "LU", "MK", "MT", "MD", "NL", "NO", "PL", "PT", "RO", "RU", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "RS", "ME"]
    for country in countries:
        global COUNTRIES
        COUNTRIES = [country]
        g = read_topology_networkx()
        if not 20 < g.number_of_nodes() < 1500:
            continue
        pre_calculations(g)
        # 50 times, choose 5 random probes and 1 random target
        for i in range(50):
            CR = np.random.choice(list(g.nodes()), 5, replace=False)
            PH = np.random.choice(list(g.nodes()), 1, replace=False)
            # check the impact of 1 random cr 20 times, then 2 random crs 20 times, etc. until 10 random crs
            for j in range(1,11):
                for z in range(20):
                    PP = np.random.choice(list(g.nodes()), j, replace=False)
                    calculate_DIPS(g, CR, PP, PH)
                    impact = AverageImpact(g, PP, PH[0])
                    if j in impacts:
                        impacts[j].append(impact)
                    else:
                        impacts[j] = [impact]
    # plot the average impact of each number of crs
    x = []
    y = []
    for key in impacts:
        x.append(key)
        y.append(sum(impacts[key])/len(impacts[key]))
    plt.plot(x,y)
    plt.xlabel("Number of Probes")
    plt.ylabel("Average Impact")
    plt.savefig("probes_vs_impact.png", dpi=1000)
    plt.close()

'''
    Validates that using a cooperative host increases the impact on the target
    Plots the average impact of using cooperative and non cooperative hosts
'''
def validate_cooperative_cr():
    impactsNonCooperative = {}
    impactsCooperative = {}
    # create a list of all countries in europe
    countries = ["PH","PL", "GB", "AL", "AD", "AM", "AT", "AZ", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FO", "FI", "FR", "GE", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI", "LT", "LU", "MK", "MT", "MD", "NL", "NO", "PL", "PT", "RO", "RU", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "RS", "ME"]
    for country in countries:
        global COUNTRIES
        global COOPERATIVE
        COUNTRIES = [country]
        g = read_topology_networkx()
        if not 20 < g.number_of_nodes() < 1500:
            continue
        pre_calculations(g)
        # 50 times, choose 5 random probes and 1 random target
        for i in range(50):
            PP = np.random.choice(list(g.nodes()), 5, replace=False)
            PH = np.random.choice(list(g.nodes()), 1, replace=False)
            # check the impact of 1 random cr 20 times, then 2 random crs 20 times, etc. until 10 random crs
            for j in range(1,11):
                for z in range(20):
                    CR = np.random.choice(list(g.nodes()), j, replace=False)
                    COOPERATIVE = False
                    calculate_DIPS(g, CR, PP, PH)
                    impact = AverageImpact(g, PP, PH[0])
                    if j in impactsNonCooperative:
                        impactsNonCooperative[j].append(impact)
                    else:
                        impactsNonCooperative[j] = [impact]
                    COOPERATIVE = True
                    calculate_DIPS(g, CR, PP, PH)
                    impact = AverageImpact(g, PP, PH[0])
                    if j in impactsCooperative:
                        impactsCooperative[j].append(impact)
                    else:
                        impactsCooperative[j] = [impact]
 
    # plot the average impact of both cooperative and non cooperative crs for each number of crs in the same plot
    x = []
    y = []
    for key in impactsNonCooperative:
        x.append(key)
        y.append(sum(impactsNonCooperative[key])/len(impactsNonCooperative[key]))
    y2 = []
    for key in impactsCooperative:
        y2.append(sum(impactsCooperative[key])/len(impactsCooperative[key]))
    plt.plot(x,y, label="Non Cooperative")
    plt.plot(x,y2, label="Cooperative")
    plt.xlabel("Number of CRs")
    plt.ylabel("Average Impact")
    plt.legend()
    plt.savefig("cooperative_vs_non_cooperative.png", dpi=1000)
    plt.close()
 
'''
    Calculates the expected value of successfully manipulating within the same state as the host versus in another state, 
    based on the number of CRs, with a cooperative host in the US
'''
def check_state_manipulation():
    global COUNTRIES
    global COOPERATIVE
    COOPERATIVE = True
    COUNTRIES = ["US"]
    g = read_topology_networkx()
    pre_calculations(g)
    iterations = 1000
    average_per_cr = {}
    for cr in range(6):
        total_different = 0
        total_same = 0
        crs = get_n_best_cr_v3(g,cr)
        for iter in range(iterations):
            chance_same_state = 1
            count_same_state = 0
            chance_different_state = 1
            count_different_state = 0
            PP = np.random.choice(list(g.nodes()), 5, replace=False)
            PH = np.random.choice(list(g.nodes()), 1, replace=False)
            host_state = ""
            for state in states_to_names.keys():
                if PH in states_to_names[state]:
                    host_state = state
                    break
            calculate_DIPS(g,crs,PP,PH)
            for v in g.nodes():
                if not v in states_to_names[host_state]:
                    chance_different_state += DiPGroup(g,PP,PH[0],v)
                    count_different_state += 1
                else:
                    chance_same_state += DiPGroup(g,PP,PH[0],v)
                    count_same_state += 1
            total_different += chance_different_state / count_different_state
            total_same += chance_same_state / count_same_state
        average_per_cr.update({cr : (total_different/iterations,total_same/iterations)})
    
    keys = list(average_per_cr.keys())
    first_values = [value[0] for value in average_per_cr.values()]
    second_values = [value[1] for value in average_per_cr.values()]

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.plot(keys, first_values, label='Different State', marker='o')
    plt.plot(keys, second_values, label='Same State', marker='o')

    plt.xlabel('CR count')
    plt.ylabel('Probability To Succeed')
    plt.title('Average Manipulation Success Rate As a Function of CR Count')
    plt.legend()

    plt.savefig("average_per_cr_noncoop.png")
        


# Example of plotting the disguise probabilty per distance like shown in the paper
def main():
    global COUNTRIES
    global COOPERATIVE
    COUNTRIES = EUROPE
    COOPERATIVE = False
    # find and check the best crs for the given countries
    probsByDistance = check_crs_for_each_country()
    print(probsByDistance)
    # sets the probability of disguise for distance 0 to 100 - sum of all other probabilities
    if(0 not in probsByDistance):
        probsByDistance[0] = 100 - sum(probsByDistance.values())
    else:
        probsByDistance[0] += 100 - sum(probsByDistance.values())+probsByDistance[0]
    

    sortedKeys = sorted(probsByDistance.keys())
    probsByDistanceSum = {key : sum([probsByDistance[k] for k in sortedKeys if k >= key]) for key in sortedKeys}
    for i in range(0,max(probsByDistanceSum.keys()),100):
        if i not in probsByDistanceSum:
            for j in sortedKeys:
                if j > i:
                    probsByDistanceSum[i] = probsByDistanceSum[j]
                    break
    plt.figure(figsize=(20,10))
    plt.bar(probsByDistanceSum.keys(), probsByDistanceSum.values(), width=80)
    plt.xticks(list(probsByDistanceSum.keys()), rotation = 90,fontsize =20)
    plt.xlabel("Distance",fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel("Probability", fontsize = 20)
    plt.savefig("probabilityPerDistanceSumEURoutersOnly.png")
    plt.close()
if __name__ == "__main__":
    main()
