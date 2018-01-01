import numpy as np
from collections import defaultdict
from graph_traversals.gt import *

file_path = "adjList.txt"

def main():
	V = 4
	# flag = input("Input from file or by input (F/I) >> ")
	adjMat = genAdjList(V) # if flag.lower() == 'f' else genAdjListFromInput() 
	DFS = DepthFisrtSearch(adjMat); DFS.dfs(); DFS.result()
	BFS = BreadthFirstSearch(adjMat); BFS.bfs(); BFS.result()
	#TPS = TopologicalSort(adjMat); TPS.tps(); TPS.result()
	#kMST = KruskalMST(adjMat); kMST.mst(); kMST.result()
	#DSP = DijkstraSP(adjMat); DSP.dsp(); DSP.result()

class Graph(object):
	# I try to well define the node so as to make a weighted directed single linked
	# it will contains it's next node and its weight to anothers.
	def __init__(self):
		self.matrix = None
		self.Graph = defaultdict(list)
		self.wGraph= defaultdict(list)
		# self.wGraph= list()
		self.vertices = None
		self.indegreeGraph = defaultdict(list)
		self.indegree = dict()

	def fit(self, matrix, vertices):
		self.matrix = matrix
		self.vertices = vertices
		self.convetToAdjList()
		self.matrix = None

	def addEdge(self, u, v, w=None):
		self.Graph[u].append(v)
		if w != None:
			self.wGraph[u].append([u, v, w])
			# self.wGraph.append([u, v, w])
		self.indegreeGraph[v].append(u)

	def countInDegree(self):
		vertices = np.arange(self.vertices)
		for key in vertices:
			if key in self.indegreeGraph.keys():
				self.indegree.update({key:len(self.indegreeGraph[key])})
			else:
				self.indegree.update({key:0})
		self.indegreeGraph = None

	def convetToAdjList(self):
		for line in self.matrix:
			self.addEdge(line[0], line[1], line[2])
		self.countInDegree()

def read_file(file_path):
	try:
		return np.loadtxt(file_path, delimiter=",", ndmin=2, dtype=np.dtype([('A', 'i4'), ('B', 'i4'), ('Weight', 'f4')]))[0]
	except:
		return np.loadtxt(file_path, delimiter=",", ndmin=2, dtype=np.dtype([('A', 'S1'), ('B', 'S1'), ('Weight', 'f4')]))[0]

def genAdjList(V):

	adjMat = Graph()
	# adjMat.fit(read_file(file_path), V)
	adjMat.fit(np.genfromtxt(file_path, delimiter=",", dtype=None), V)
	return adjMat

def genAdjListFromInput():
	vertices = int(input("Please enter the vertices of this Graph >> "))
	adjMat = Graph()
	flag = int(input("How many connections you ganna insert >> "))
	wflag = input("Are you going to insert the weight? (Y/N) >> ")
	if flag > 0 and vertices > 0:
		for i in range(flag):
			u = int(input("[{0}] The first point >> ".format(i)))
			v = int(input("[{0}]The second point >> ".format(i)))
			if wflag.lower() == 'y':
				w = float(input("[{0}]The weight is >> ".format(i)))
				print("{0} -- {1} --> {2}".format(u, w, v))
				adjMat.addEdge(u, v, w)
			else: 
				print("{0} ----> {1}".format(u, v))
				adjMat.addEdge(u, v)
		adjMat.vertices = vertices
		adjMat.countInDegree()
	else:
		raise ValueError
	return adjMat

if __name__ == '__main__':
	main()