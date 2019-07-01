import numpy as np
import copy
import time
import heapq
import random
from queue import Queue

MAX = np.iinfo(np.int32).max


def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')


class Node:
    def __init__(self, id):
        self.id = id


class Edge:
    def __init__(self, id, start, end, cost):
        self.id = id
        self.start = start
        self.end = end
        self.cost = cost

    def update(self, id, start, end, cost):
        self.id = id
        self.start = start
        self.end = end
        self.cost = cost


class Tree:
    def __init__(self, nodes=set(), edges=set()):
        self.nodes = nodes
        self.edges = edges

    def add_node(self, node):
        for vertex in self.nodes:
            if vertex.id == node.id:
                raise Exception('Node already there')
        self.nodes.add(node)

    def add_edge(self, edge):
        if edge.start in self.nodes and edge.end in self.nodes:
            for arc in self.edges:
                if arc.start == edge.start and arc.end == edge.end or arc.id == edge.id:
                    raise Exception('Edge already exists')
            self.edges.add(edge)
        else:
            raise Exception("Edge not present in nodes")


class Solution:
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0  # current cost
        self.h = 0  # heuristic
        self.graph = graph
        self.visited = [places[0]]  # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:])  # list of attractions not yet visited

    def __lt__(self, other):
        return self.g + self.h < other.g + other.h

    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        node = self.not_visited[idx]
        last_node = self.visited[-1]

        self.visited.append(node)
        self.not_visited.remove(node)
        self.g += self.graph[last_node, node]

    def swap(self, id1, id2):
        """
        Swaps indices and updates solution
        """
        temp_node = self.visited[id1]
        self.visited[id1] = self.visited[id2]
        self.visited[id2] = temp_node

        self.g = 0
        for idx in range(0, len(self.visited) -1):
            self.g += self.graph[self.visited[idx], self.visited[idx + 1]]


def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    solution = Solution(places, graph)
    return_sol = copy.deepcopy(solution)
    return_sol.g = np.iinfo(np.int32).max
    solutions = Queue()
    solutions.put(solution)
    while not solutions.empty():
        current_solution = solutions.get()

        if len(current_solution.visited) == len(places) - 1:
            current_solution.add(0)
            solutions.put(current_solution)
        elif len(current_solution.visited) == len(places) and current_solution.g < return_sol.g:
            return_sol = current_solution
        else:
            for idx in range(0, len(current_solution.not_visited) - 1):
                new_sol = copy.deepcopy(current_solution)
                new_sol.add(idx)
                solutions.put(new_sol)

    return return_sol

   

def initial_sol(graph, places):
    """
    Return a completed initial solution
    """
    return dfs(graph, places)


def dfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    solution = Solution(places, graph)
    return_sol = copy.deepcopy(solution)
    return_sol.g = np.iinfo(np.int32).max
    solutions = list()
    solutions.append(solution)
    while len(solutions) > 0:
        current_solution = solutions.pop()

        if len(current_solution.visited) == len(places) - 1:
            current_solution.add(0)
            solutions.append(current_solution)
        elif len(current_solution.visited) == len(places):
            return current_solution
        else:
            temp_list = list()
            for idx in range(0, len(current_solution.not_visited) - 1):
                temp_list.append(idx)
            random.shuffle(temp_list)
            for idx in temp_list:
                new_sol = copy.deepcopy(current_solution)
                new_sol.add(idx)
                solutions.append(new_sol)

    raise Exception('Cuz I''m in too deep - Sum41')

def shaking(sol, k):
    """
    Returns a solution on the k-th neighrboohood of sol
    """
    nb_swaps = 0
    solutions = []
    for nb_swaps in range (0,k):
        max_id = len(sol.visited) - 2
        id1 = random.randint(1, max_id)
        id2 = -1
        while id2 == -1 or id2 == id1:
            id2 = random.randint(1, max_id)
        new_sol = copy.deepcopy(sol)
        new_sol.swap(id1, id2)
        solutions.append(new_sol)
        sol = new_sol
    
    return solutions.pop()
    
def local_search_2opt(sol):
    """
    Apply 2-opt local search over sol
    """
    new_sol = copy.deepcopy(sol)
    profitable_swap_found = True
    while profitable_swap_found:
        profitable_swap_found = False
        for id1 in range (1, len(sol.visited) - 3):
            for id2 in range(id1 + 2, len(sol.visited) - 1):
                if id1 != id2:
                    temp_sol = copy.deepcopy(new_sol)
                    temp_sol.swap(id1, id2)
                    if temp_sol.g < new_sol.g:
                        new_sol = temp_sol
                        profitable_swap_found = True
                        
    return new_sol
    
def vns(sol, k_max, t_max):
    """
    Performs the VNS algorithm
    """
    start_time = time.time()  
    current_best_sol = sol
    while t_max > time.time() - start_time:
        new_sol = shaking(current_best_sol, k_max)
        new_sol = local_search_2opt(new_sol)
        if new_sol.g < current_best_sol.g:
            current_best_sol = new_sol
            
    return current_best_sol
        
    
def fastest_path_estimation(sol):
    """
    Returns the time spent on the fastest path between
    the current vertex c and the ending vertex pm
    """
    c = sol.visited[-1]
    if not sol.not_visited:
        return 0
    pm = sol.not_visited[-1]
    nodes_to_visit = set(sol.not_visited)
    distances = {}
    for node in nodes_to_visit:
        distances[node] = np.iinfo(np.int32).max
    nodes_to_visit.add(c)
    distances[c] = 0
    while len(nodes_to_visit) > 0:
        min_node = min(nodes_to_visit, key=lambda k: distances[k])
        if min_node == pm:
            return distances[pm]
        nodes_to_visit.remove(min_node)
        for node in nodes_to_visit:
            dist = distances[min_node] + sol.graph[min_node, node]
            if dist < distances[node]:
                distances[node] = dist
                
    raise Exception("Dijkstra has failed us :( XD ...")


def A_star(graph, places, dijkstra):
    """
    Performs the A* algorithm
    """
    # blank solution
    root = Solution(graph=graph, places=places)

    # search tree T
    T = []
    heapq.heapify(T)
    heapq.heappush(T, root)
    while len(T) > 0:
        current_solution = heapq.heappop(T)

        if len(current_solution.visited) == len(places) - 1:
            current_solution.add(0)
            heapq.heappush(T, current_solution)
        elif len(current_solution.visited) == len(places) :
            return current_solution
        else:
            for idx in range(0, len(current_solution.not_visited) - 1):
                new_sol = copy.deepcopy(current_solution)
                new_sol.add(idx)
                # Dijsktra
                if dijkstra:
                    new_sol.h = fastest_path_estimation(new_sol)
                # Edmonds
                else:
                    new_sol.h = minimum_spanning_arborescence(new_sol)
                heapq.heappush(T, new_sol)

    raise Exception("A_star has failed us :( XD ...")


def minimum_spanning_arborescence(sol):
    root = sol.visited[-1]
    nodes = set()
    edges = set()
    idx = 0
    vertexes = set()
    for vertex in sol.not_visited:
        found = False
        for vert in vertexes:
            if vert.id == vertex:
                found = True
        if not found:
            vertexes.add(Node(vertex))

    root_in_vertices = False
    for vert in vertexes:
        if vert.id == root:
            root_in_vertices = True
    if not root_in_vertices:
        vertexes.add(Node(root))
    for start_vertex in vertexes:
        for end_vertex in vertexes:
            if start_vertex != end_vertex and end_vertex.id != root:
                edge = Edge(idx, start_vertex, end_vertex, sol.graph[start_vertex.id, end_vertex.id])
                edges.add(edge)
                idx += 1
        nodes.add(start_vertex)
    tree = Tree(nodes, edges)
    id_vertex = max(vertexes, key=lambda x: x.id).id + 1
    id_edge = idx

    min_span_tree = recursive_edmonds(tree, root, id_vertex, id_edge)
    
    cost = 0
    for edge in min_span_tree.edges:
        cost += edge.cost
        
    return cost


def recursive_edmonds(tree, root, id_vertex, id_edge):
    best_in_edges = {}
    kicks_out = {}
    real = {}
    for vertex in tree.nodes:
        if vertex.id != root:
            min_edge = None
            min_cost = MAX
            for edge in tree.edges:
                if edge.end == vertex and edge.cost < min_cost:
                    min_cost = edge.cost
                    min_edge = edge
            best_in_edges[vertex] = min_edge
            
            # look for cycle
            start_node = vertex
            cycle = [start_node]
            cycle_edges = []
            cycle_found = False
            while start_node is not None and not cycle_found:
                from_node = best_in_edges[start_node].start
                if from_node in best_in_edges:
                    cycle_edges.append(best_in_edges[start_node])
                    start_node = from_node
                    cycle.append(start_node)
                else:
                    start_node = None
                if start_node == vertex:
                    cycle_found = True
                    cycle.pop()
                   
            if cycle_found:
                new_vertex = Node(id_vertex)
                id_vertex += 1
                subtree = Tree(set(), set())
                for node in tree.nodes:
                    if node not in cycle:
                        subtree.add_node(node)
                subtree.add_node(new_vertex)
                sub_edges = set()
                for edge in tree.edges:
                    new_edge = Edge(0, 0, 0, 0)
                    if edge.start not in cycle and edge.end not in cycle:
                        new_edge.update(id_edge, edge.start, edge.end, edge.cost)
                        id_edge += 1
                        real[new_edge] = edge
                        sub_edges.add(new_edge)
                    elif edge.start in cycle and edge.end not in cycle:
                        new_edge.update(id_edge, new_vertex, edge.end, edge.cost)
                        id_edge += 1
                        real[new_edge] = edge
                        sub_edges.add(new_edge)
                    elif edge.start not in cycle and edge.end in cycle:
                        new_edge.update(id_edge, edge.start, new_vertex, edge.cost)
                        kicks_out[new_edge] = best_in_edges[edge.end]
                        new_edge.cost -= best_in_edges[edge.end].cost
                        id_edge += 1
                        real[new_edge] = edge
                        sub_edges.add(new_edge)

                subtree.edges = sub_edges
                arborescence = recursive_edmonds(subtree, root, id_vertex, id_edge)
                
                return_tree = Tree(set(), set())
                
                kicked_out = None
                for edge in arborescence.edges:
                    if edge in real:
                        real_edge = real[edge]
                        if real_edge.start not in return_tree.nodes:
                            return_tree.add_node(real_edge.start)
                        if real_edge.end not in return_tree.nodes:
                            return_tree.add_node(real_edge.end)
                        if real_edge not in return_tree.edges:
                            return_tree.add_edge(real_edge)
                    if edge.end == new_vertex:
                        kicked_out = kicks_out[edge]
                for edge in tree.edges:
                    if edge in cycle_edges and edge != kicked_out:
                        if edge.start not in return_tree.nodes:
                            return_tree.add_node(edge.start)
                        if edge.end not in return_tree.nodes:
                            return_tree.add_node(edge.end)
                        return_tree.add_edge(edge)
                return return_tree
        
    cycle_free_tree = Tree(set(), set())
    for node, in_edge in best_in_edges.items():
        if node not in cycle_free_tree.nodes:
            cycle_free_tree.add_node(node)
        if in_edge.start not in cycle_free_tree.nodes:
            cycle_free_tree.add_node(in_edge.start)
        cycle_free_tree.add_edge(in_edge)
    return cycle_free_tree

def main():
    graph = read_graph()
    
    print("***** BFS *****")
    #test 1  --------------  OPT. SOL. = 27
    start_time = time.time()
    places=[0, 5, 13, 16, 6, 9, 4]
    sol = bfs(graph=graph, places=places)
    print(sol.g)
    print(sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    #test 2 -------------- OPT. SOL. = 30
    start_time = time.time()
    places=[0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
    sol = bfs(graph=graph, places=places)
    print(sol.g)
    print(sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    #test 3 -------------- OPT. SOL. = 26
    start_time = time.time()
    places=[0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
    sol = bfs(graph=graph, places=places)
    print(sol.g)
    print(sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("\n***** A_star DIJ *****")
    # test 1  --------------  OPT. SOL. = 27
    start_time = time.time()
    places = [0, 5, 13, 16, 6, 9, 4]
    astar_sol = A_star(graph=graph, places=places, dijkstra=True)
    print(astar_sol.g)
    print(astar_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    # test 2  --------------  OPT. SOL. = 30
    start_time = time.time()
    places = [0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
    astar_sol = A_star(graph=graph, places=places, dijkstra=True)
    print(astar_sol.g)
    print(astar_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    # test 3  --------------  OPT. SOL. = 26
    start_time = time.time()
    places = [0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
    astar_sol = A_star(graph=graph, places=places, dijkstra=True)
    print(astar_sol.g)
    print(astar_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    # test 4  --------------  OPT. SOL. = 40
    start_time = time.time()
    places = [0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
    astar_sol = A_star(graph=graph, places=places, dijkstra=True)
    print(astar_sol.g)
    print(astar_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("\n***** A_star EDM *****")
    # test 1  --------------  OPT. SOL. = 27
    start_time = time.time()
    places = [0, 5, 13, 16, 6, 9, 4]
    astar_sol = A_star(graph=graph, places=places, dijkstra=False)
    print(astar_sol.g)
    print(astar_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    # test 2  --------------  OPT. SOL. = 30
    start_time = time.time()
    places = [0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
    astar_sol = A_star(graph=graph, places=places, dijkstra=False)
    print(astar_sol.g)
    print(astar_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    # test 3  --------------  OPT. SOL. = 26
    start_time = time.time()
    places = [0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
    astar_sol = A_star(graph=graph, places=places, dijkstra=False)
    print(astar_sol.g)
    print(astar_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    # test 4  --------------  OPT. SOL. = 40
    start_time = time.time()
    places = [0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
    astar_sol = A_star(graph=graph, places=places, dijkstra=False)
    print(astar_sol.g)
    print(astar_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("\n***** VNS *****")
    # test 1  --------------  OPT. SOL. = 27
    places=[0, 5, 13, 16, 6, 9, 4]
    sol = initial_sol(graph=graph, places=places)
    start_time = time.time()
    vns_sol = vns(sol=sol, k_max=10, t_max=1)
    print(vns_sol.g)
    print(vns_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    #test 2  --------------  OPT. SOL. = 30
    places=[0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
    sol = initial_sol(graph=graph, places=places)

    start_time = time.time()
    vns_sol = vns(sol=sol, k_max=10, t_max=1)
    print(vns_sol.g)
    print(vns_sol.visited)

    print("--- %s seconds ---" % (time.time() - start_time))
    # test 3  --------------  OPT. SOL. = 26
    places=[0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
    sol = initial_sol(graph=graph, places=places)

    start_time = time.time()
    vns_sol = vns(sol=sol, k_max=10, t_max=1)
    print(vns_sol.g)
    print(vns_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))
    # test 4  --------------  OPT. SOL. = 40
    places=[0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
    sol = initial_sol(graph=graph, places=places)

    start_time = time.time()
    vns_sol = vns(sol=sol, k_max=10, t_max=1)
    print(vns_sol.g)
    print(vns_sol.visited)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
