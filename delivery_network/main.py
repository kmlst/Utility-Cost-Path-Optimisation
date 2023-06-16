
from graph import Graph
import collections as c
import bisect
import random
import numpy as np
import matplotlib.pyplot as plt
data_path = "input/"
file_name = "network.01.in"

#g = Graph.graph_from_file(data_path + file_name)
g1 = Graph.graph_from_file("input/network.00.in")
#print(g)
#graph_render(g1,path=[9,8,1,2,3,4,10])
#time_measure(graph_render(g1,path=[9,8,1,2,3,4,10]))




def truck_from_file(i):
    """
    Reads a text file containing truck information 
    and returns a list containing all useful trucks, sorted
    in decreasing order of both power and cost.
    The file should have the following format: 
        The first line of the file is 'n' the number of trucks
        The next n lines have 'power cost'
        All values are integers.

    Parameters:
    ------------
        i : int
            Number of the truck file

    Output:
    ----------
    trucks[cost, power] : list[int, int]
    
    with
        cost : int
            The cost of the truck
        power: int
            The power of the truck
    The list is sorted in decreasing order of both power and cost.
    """

    with open(f"input/trucks.{i}.in", 'r') as f:
        f.readline()
        all_trucks = {} # power:int -> cost:int
        for truck in f.readlines():
            cost, power = truck.split()
            cost, power = map(int, (cost, power))
            if power in all_trucks: #only keeping smallest cost for each power
                all_trucks[power] = min(all_trucks[power], cost)
            else:
                all_trucks[power] = cost
    
    all_trucks = list(all_trucks.items())
    # sorting in decreasing order of power
    all_trucks.sort(key=lambda x:x[0], reverse=True)
    trucks = [all_trucks[0]]
    
    previous_cost = all_trucks[0][1]
    for power, cost in all_trucks:
        #not adding trucks with stricly less power costing more
        if cost < previous_cost:
            trucks.append((power, cost))
            previous_cost = cost
    return trucks


def route_proccessing(i, trucks, filewrite=False):
    """
        Computes the cost of each route in routes.i.out and 
        outputs a list of routes with associated cost sorted by 
        utility/cost. NodeType is assumed to be int.
        
        Optionally if filewrite is True, 
        writes a file named routes.processed.i.in  with format:
            Each line is of the form 'a b power utility cost utility/cost'
            where power is that of the truck paired to the route.
            power, utility, cost are integers ; a, b are node tags.

        Parameters: 
        -----------
            i: int
                The number of the truck file
            filewrite: bool
                indicates if a file containing the result is written
            trucks : list[(int, int)]
                Entries represent the couple (power, cost) for a truck

        Outputs: 
        -----------
            routes : list[NodeType, NodeType, int, int, int, float]
                Each tupple in the list indicates in order: the two nodes of the path,
                the power of the truck, the cost of the truck, the utility gained from the path and the ratio of utility/cost
    """
    n_trucks = len(trucks)
    routes = []
    with open(f"output/routes.{i}.out", 'r') as f:
        f.readline()
        for line in f.readlines(): # readlines is a generator
            a, b, utility, pmin = map(int, line.split(" ")) 
            # cost is the closest key, on the right
            ind = bisect.bisect_right(trucks, pmin, key=lambda x:x[0])
            if ind < n_trucks-1: # route with too great power necessary
                cost = trucks[ind][1]
                routes.append((a, b, trucks[ind][0], utility, cost, utility/cost))
    #sorting on ratio utility cost
    routes.sort(key=lambda x : x[5], reverse=True)
    if filewrite:
        processed = open(f"input/routes.processed.{i}.out", 'w')
        for a, b, pmin, utility, cost, u_c in routes:
            processed.write(f"{a} {b} {pmin} {utility} {cost} {u_c}\n")
        processed.close()
    return routes


def route_out(i):
    """
        Writes a file routes.i.out based on routes.i.in .
        The first line of routes.i.in has format 'n' where n is the number of routes in the file
        The n following lines should have format: 'Node1 Node2 utility'
            with        utility  : int
                    Node1, Node2 : Nodetype
        Output file has the same first line, and the 
        n following lines are in format 'Node1 Node2 utility power_min'
            with       power_min : int

        Parameters
        ----------------
            i : int
                number of the file to convert

        Output
        ---------
            NoneType
    """

    G = Graph.graph_from_file(f"input/network.{i}.in").kruskal()

    with open(f"input/routes.{i}.in") as f:
        output = open(f"output/routes.{i}.out", 'w')
        N = int(f.readline())
        output.write(f"{N}\n")
        for ligne in f.readlines():
            a, b, utility = ligne.split()
            power_min = int(G.min_power(int(a),int(b))[1])
            output.write(a + " " + b + " " + utility + " " + str(power_min)+"\n")
        output.close()


def simulated_annealing(trucks, routes):
    """
    Implements a simulated annealing heuristic to maximize utility.

    Parameters
    -------------
      trucks : list[(int, int)]
        List of useful trucks sorted in decreasing order of both power and cost
        Entries are tuples representing in order: (power, cost)

      routes : list[NodeType, NodeType, int, int, float]
                Each tupple in the list indicates in order: the two nodes of the path,
                the power of the associated truck, its' cost, the utility gained from the path
                 and the ratio of utility/cost. It is sorted in descending utility/cost

    Output
    -----------
      best_chosen_routes: list[NodeType, NodeType, int, int, float]
          Represents the choice of routes giving the maximum utility among 
          those explored. Each route is associated to an unique truck by its cost.
      utility: int
          The maximum utility observed during exploration
    """
    # choosing an inital point,
    # can be random or deterministic 
    # here we start from a greedy approach
    budget = 25 * 10**9
    best_chosen_routes = {}
    max_utility = 0
    min_cost = routes[-1][3]
    for path in reversed(routes):
        a, b, power, cost, utility, u_c = path
        if cost < budget:
            best_chosen_routes[path] = 1
            budget -= cost
        if budget < min_cost:
            break
        else:
            pass
    
    def utility(road_choices):
      total = 0
      for _, _, _, _, u, _ in road_choices:
        total += u
      return total    

    historique = []
    chosen_routes = best_chosen_routes.copy()
    T = 10**6 #Temperature
    K_activation = 10**3 # hyperparameter constant
    current_u = utility(chosen_routes)

    while T > 10**2:
        
        if current_u > max_utility:
            max_utility = current_u
            best_chosen_routes = chosen_routes.copy()
        # trying neighbouring situations

        # first trying for ameliorations when there are ones
        counter = 1
        while counter < 3: #three tries to improve 
            r_change = random.choice(routes)
            delta_u = 0
            if r_change not in chosen_routes:
                if budget - r_change[3] > 0:
                    chosen_routes[r_change] = 1
                    budget -= r_change[3]
                    delta_u = r_change[4]
            current_u += delta_u
            historique.append(current_u)
            counter += 1

        chosen_routes.pop(r_change)
        delta_u = -r_change[4]
        # accepting locally to lower utility
        if random.random() < np.exp( delta_u /(K_activation * T)):
            budget += r_change[3]
        else:
            #reverse change
            delta_u = 0
            chosen_routes[r_change] = 1

        current_u += delta_u
        historique.append(current_u)
        T *= 0.995

    N =len(historique)
    X = np.arange(N)
    plt.plot(X, historique, linewidth = 1, color='blue')
    plt.plot(X, np.ones(N)*max_utility, linewidth = 1, color='red')
    font1 = {'family':'serif','color':'blue','size':20}
    font2 = {'family':'serif','color':'red','size':20}
    font2 = {'family':'serif','color':'red','size':20}
    plt.xlabel("temps", fontdict = font1)
    plt.ylabel("utility", fontdict = font2)
    plt.title("Exploration of utility levels by simulated annealing")
    plt.show()

    return best_chosen_routes, max_utility, historique

if __name__ == '___main___':
    main()


#print(truck_from_file(1))

#print(route_proccessing(1, truck_from_file(1)))
truck = truck_from_file(1)
routes = route_proccessing(1, truck)
# solution optimale pour les petits grapghe
#print(simulated_annealing(truck, routes))
#print(sum(r[4] for r in routes))
_, max_utility, historique = simulated_annealing(truck, routes)


N =len(historique)
X = np.arange(N)
plt.plot(X, historique, linewidth = 1, color='blue')
plt.plot(X, np.ones(N)*max_utility, linewidth = 1, color='red')
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'red','size':20}
font3 = {'family':'serif','color':'purple','size':16}
plt.xlabel("temps", fontdict = font1)
plt.ylabel("utility", fontdict = font2)
plt.title("Exploration of utility levels by simulated annealing", fontdict=font3)
plt.savefig('image.pdf')
