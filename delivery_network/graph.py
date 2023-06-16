import graphviz as gph
from time import perf_counter
import numpy as np


class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes]) # n1 -> (n2, power, distance)
        self.nb_nodes = len(nodes)
        self.nb_edges = 0


    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output


    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        #checking if nodes are in the graph, adding them if not
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []
        # edge addition
        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1


    def get_path_with_power(self, src, dest, power):
        """
        Returns an admissible path from src to dest with given power if possible
        and returns None otherwise. 
        Returned path isn't necessarily optimal in the sense of distance;

        Parameters: 
        -----------
        src: NodeType
            First end (node) of the edge
        dest: NodeType
            Second end (node) of the edge
        power: numeric (int or float)
            power of the agent

        Output
        ----------
        path : list[NodeType]
            Sequence of nodes leading from src to dest through edges
            whose power is less than that of the agent.
        """
        # Deep First Search through recursion

        seen = {} # edge (a, b) : True

        def rec_path(position):
            if position == src: 
                return [src]

            neighbors = self.graph[position]
            # Moving to neighboors
            for n in neighbors:  # n = (node tag, power, dist)
                if (position, n[0]) not in seen:
                    seen[(position, n[0])] = True 
                    seen[(n[0], position)] = True 
                    if n[1] <= power: # moving through only admissible paths
                        p = rec_path(n[0]) 
                        if p:
                            p.append(position) 
                            return p
        path = rec_path(dest)
        return path

    def connected_components(self): 
        """
        Returns the connected components of the graph in the form 
        of a list of lists.  
        
        Output
        -----------
        list_of_components : list[list[int]]
            Each sublist represents a connected component.
            The sublist contains the tags of all nodes in the component
        """
        seen = {n: False for n in self.nodes} 
        list_of_components = []

        for a in self.nodes:
            if not seen[a]: # if a wasn't seen, a belongs to a yet unseen connected component
                seen[a] = True
                connected_component = [a] 

                to_add = []
                for b in self.graph[a]:# heap of nodes to add is just neighboors
                    to_add.append(b[0])
                    # not necessary to check if seen, as none of the nodes
                    # in this connected component were seen (a would've also been otherwise)
                
                while to_add:
                    b = to_add.pop()
                    connected_component.append(b)
                    seen[b] = True
                    for b_neighboor in self.graph[b]:
                        if not seen[b_neighboor[0]]:
                            # have to check if seen 
                            # (e.g. neighboor of b being neighboor of a already seen before)
                            to_add.append(b_neighboor[0])
                list_of_components.append(connected_component)
        return list_of_components

    def connected_components_set(self):
        """
        Returns the connected components of a graph 

        Output: 
        ----------
        Type set[frozenset[NodeType]]
            set of connected component, 
            each being represented as a frozen set
        """
        return set(map(frozenset, self.connected_components()))


    def min_power(self, src, dest):
        """
        Returns path from src to dest with minimal power and the associated power when there is one
        Returns None otherwise

        Parameters:
        -----------
        src: NodeType
            The source node
        dest: NodeType
            The destination node
        
        Output: 
        --------
        tupple(list[Nodetype], float)  | NoneType
                    
        with
                path : list[Nodetype]
                    Sequence of nodes leading from src to dest through edges
                    whose power is less than that of the agent.

                power : float
                    Minimal power necessary to go from src to dest
        """
        #constructing list of distinct powers
        powers = [] 
        for n in self.nodes:
            for e in self.graph[n]:
                powers.append(e[1])

        powers = list(set(powers))
        powers.sort() #in place
        # Dichotomic research
        i = 0
        j = len(powers) - 1
        while i < j:
            result_j = self.get_path_with_power(src, dest, powers[j])
            if result_j is None:
                return None
            result_midway = self.get_path_with_power(src, dest, powers[(i+j)//2])
            if result_midway is not None:
                j = (i+j)//2
            else:
                if j == i+1:
                    i = j
                else:
                    i = (i+j)//2
        return self.get_path_with_power(src, dest, powers[i]), powers[i]
            

    def pmin(edges):
            """
            Returns the index of the edge with minimal power in the list edges
            """

    def kruskal(self):
        """
        Returns the minimal covering tree of the graph.
        Construction uses Kruskal's algorithm.

        Output
        ---------
        G : Graph
        """
        # other idea is to create a subclass that inherits from Graph
        #Tree would have attribute parents and descendants

        # constructing list of (power, edge) without repetition
        edges_seen = {} #to check repetition
        edges = []
        for node_a in self.nodes:
            for edge in self.graph[node_a]: 
                node_b, p, d = edge
                if not ((node_a, node_b) in edges_seen):
                    edges_seen[(node_a, node_b)] = True
                    edges_seen[(node_b, node_a)] = True # to not add it again when on node_b
                    edges.append((node_a, node_b, p, d))
        # sorting in place on powers
        edges.sort(key= lambda a : a[2]) 

        #constructing covering tree 
        G = Graph(self.nodes)
        connected = {n: [n] for n in self.nodes} #to identify accessible nodes from n
        # using dictionary for O(1) check if two nodes are connected 
        for edge in edges:
            # loop invariant: connected[n] contains the connected component of node n at the end of each iteration
            node_a, node_b, p, d = edge
            if not (connected[node_b][0] == connected[node_a][0]):
                G.add_edge(node_a, node_b, p, d)
                for node_c in connected[node_b]: # none of the nodes connected to b were connected to a
                    connected[node_a].append(node_c)
                    connected[node_c] = connected[node_a] # pointers
                    # changing connected[a] will thus update it automatically 
                    # for all nodes in the connected component
                    
            if G.nb_edges == G.nb_nodes - 1: # optionnal
                break # trees of size V have exactly V-1 edges : if it is reached, construction's over.
        return G
        
    def min_power_tree(self, src, dest):
        """
        Returns, if there is one, the only path from src to dest in a tree and the minimal
        power necessary to cross it.
        By construction of the minimal covering tree, the power is 
        the smallest one needed to go from src to dest
        It is assumed the function is being applied on a minimal covering tree,
        e.g. the output of the kruskal function
        

        Parameters: 
        -----------
        src: NodeType
            source node
        dest: NodeType
            destination node

        Output
        ----------
        tupple(list[NodeType], float) | NoneType
        
        with
                final_path : 
                    Sequence of nodes leading from src to dest through edges
                    whose power is less than that of the agent.

                min_p : float
                    Minimal power necessary to go from src to dest
        """
        # Recursive Deep First Search
        # unlike with a graph, we can just check nodes
        seen = {} # node : True

        def rec_path(position): #keeping track of the minimal power on the path
            if position == src:
                return [(src, 0)]      
            #checking neighbors
            for edge in self.graph[position]:
                node_b, power, _ = edge
                if node_b not in seen:
                    seen[node_b] = True
                    result = rec_path(node_b)
                    if result is not None:
                        result.append((position, power))
                        return result
        path = rec_path(dest)
        min_p = 0
        final_path = []
        for node, p in path:
            if p > min_p:
                min_p = p
            final_path.append(node)
        return final_path, min_p

    # no 'self' in args of method
    @staticmethod 
    def graph_from_file(filename):
        """
        Reads a text file and returns the graph as an object of the Graph class.
        The file should have the following format: 
            The first line of the file is 'n m'
            The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
            The nodes (node1, node2) should be named 1..n
            All values are integers.

        Parameters: 
        -----------
        filename: str
            The name of the file

        Outputs: 
        -----------
        G: Graph
            An object of the class Graph with the graph from file_name.
        """

        graphe = open(filename, "r")
        first_line = graphe.readline().split(" ") #first line: number of nodes, number of edges
        n = int(first_line[0])
        G = Graph(list(range(1, n+1))) #initialization of graph
        
        E = graphe.readlines() 
        for edge in E: #filling G with specified edges
            edge = ''.join(edge.splitlines()) 
            ar = list(map(float, edge.split(" "))) # format : node_a, node_b, power, distance (optional)
            if len(ar) == 4: # distance specified
                 G.add_edge(int(ar[0]), int(ar[1]), ar[2], ar[3])
            elif len(ar) == 3: # not specified
                G.add_edge(int(ar[0]), int(ar[1]), ar[2], 1)
        graphe.close()
        return G

    def build_tree(self):
        """
        Takes a graph with the structure of a minimal covering tree, as provided
        by the kruskal function, and returns a tree structure organizing it.


        Output
        --------
        tupple(dict[NodeType : (NodeType, float)], dict[Nodetype : int], NodeType)
        with
            parents : dict[NodeType : (NodeType, float)]
                To each node associates: (parent, power) where power is the 
                By convention, a root is its' own parent. 
                There can be multiple roots, e.g. graphs with two connex components

            depth: dict[Nodetype : int]
                To each node associates corresponding depth in the tree
        """
        #choosing the root : edge with max nb_edges
        # for now we assume there is only one connected component
        # and thus one root. 
        # Other case can be dealt with a loop over connected components
        # and using this function on each 
        root = self.nodes[0]
        max_edges = len(self.graph[root])
        for n in self.nodes:
            n_edges = len(self.graph[n])
            if n_edges > max_edges:
                max_edges = n_edges
                root = n            
        # Deep first search 
        parents = {root:(root, 0, 0)}
        depths = {root:0}
        position = [root]
        def dfs(position):
            for node, p, d in self.graph[position]:
                if node != parent[position]:
                    parents[node] = (parent, p)
                    depths[node] = depths[parent]+1
                    dfs(node)           
        return parents, depths

    def min_power_optimised(self, src, dest):
        """
        Computes the minimal covering tree of the graph through kruskal's
        algorithm and then finds path with minimal power from src 
        to dest when there is one.

        Parameters
        ----------
        
        src: NodeType
            Source node
        dest: NodeType
            Destination Node

        Output
        ----------
        (path, power) : list[NodeType], float) | NoneType
        
        with
                path : list[NodeType]
                    Sequence of nodes leading from src to dest through edges
                    whose power is less than that of the agent.

                power : float
                    Minimal power necessary to go from src to dest
        """
        G = self.kruskal() #covering tree graph
        parents, depths = G.build_tree() # tree structure
        #building path from src to root 
        path_src = [(src, 0)]
        for i in range(depths[src]):
            next_parent, p = parents[path1[i]]
            path_src.append(next_parent, max(p, path_src[i][1]))
        #building path from dest to root 
        path_dest = [(dest, 0)]
        for i in range(depths[dest]):
            next_parent, p = parents[path1[i]]
            path_dest.append(next_parent, max(p, path_dest[i][1]))
        # finding lowest common ancestor
        while path_dest and path_src and path_src[-1]==path_dest[-1]:
            ancestor, p = path_src.pop()
            path_dest.pop()
        #lowest common ancestor was removed, it has to be put back
        path_src.append(ancestor, p)
        #joining the paths
        final_path = path_src # (name change)
        min_p = min(final_path, key= lambda X: X[1])
        for node, p in reversed(path_dest): # adding the dest end of the path
            final_path.append(node)
            if p < min_p:
                min_p = p
        return final_path, p




