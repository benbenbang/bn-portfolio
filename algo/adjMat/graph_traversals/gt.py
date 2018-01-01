from heapq import heappop, heappush, nsmallest, nlargest
import numpy as np

class DepthFisrtSearch(object):
	"""
	Graph should be a Class with adjList, vertices
	"""
	def __init__(self, graph):
		self.unvisited = [True] * graph.vertices
		self.graph = graph.Graph

	def dfs(self, node=0):
		self.unvisited[node] = False
		for node in self.graph[node]:
			if self.unvisited[node] is True:
				self.dfs(node)

	def result(self):
		print("DFS: Found {0} unvisited vertices".format(sum(self.unvisited)))

class BreadthFirstSearch(object):
	"""
	Graph should be a Class with adjList, vertices
	"""
	def __init__(self, graph):
		self.queue = []
		self.unvisited = [True] * graph.vertices
		self.graph = graph.Graph

	def bfs(self, start_node=0):
		self.queue.append(start_node)
		self.unvisited[start_node] = False

		while self.queue:
			node = self.queue.pop(0)
			for neighbor in self.graph[node]:
				if self.unvisited[neighbor] is True:
					self.queue.append(neighbor)
					self.unvisited[neighbor] = False

	def result(self):
		print("BFS: Found {0} unvisited vertices".format(sum(self.unvisited)))

class TopologicalSort(object):
	def __init__(self, graph):
		self.queue = []
		self.graph = graph.Graph
		self.unvisited = [True] * graph.vertices
		self.indegree = graph.indegree

	def tps(self):
		for node in range(len(self.graph)):
			if self.indegree[node] == 0:
				self.queue.append(node)
				self.unvisited[node] = False

		for node in self.queue:
			for neighbor in self.graph[node]:
				if self.unvisited[neighbor] is True and self.indegree[neighbor] > 0:
					self.indegree[neighbor] -= 1
					if self.indegree[neighbor] == 0:
						self.queue.append(neighbor)
						self.unvisited[neighbor] = False

	def result(self):
		result = self.queue if sum(self.unvisited) == 0 else "Can not find topological sort in this graph."
		print("TPS: {0}".format(result))

class KruskalMST(object):
	def __init__(self, graph):
		self.parent = [node for node in range(graph.vertices)]
		self.graph = graph.wGraph
		self.v = graph.vertices
		self.minTree = []
		self.minWeight = 0

	def find(self, i):
		if self.parent[i] == i:
			return i
		return self.find(self.parent[i])

	def union(self, u, v):
		pu = self.find(u)
		pv = self.find(v)

		if pu <= pv:
			self.parent[v] = pu
		elif pu > pv:
			self.parent[u] = pv

	def mst(self):
		i = 0
		e = 0

		# self.graph = sorted(self.graph, key=lambda x: x[2])
		self.graph = sorted([node for nodes in self.graph.values() for node in nodes],key=lambda x: x[2])

		while e != self.v-1:
			u, v, w = self.graph[i]
			i = i + 1

			pu = self.find(u)
			pv = self.find(v)

			if pu != pv:
				e += 1
				self.union(u, v)
				self.minTree.append([u, v, w])
				self.minWeight += w

	def result(self):
		print("The min spanning tree is: ")
		for i in range(len(self.minTree)):
			print("{0} --- w:{1} --- {2}".format(self.minTree[i][0], self.minTree[i][2], self.minTree[i][1]))
		print("The total cost is {0}".format(self.minWeight))

class DijkstraSP(object):
	"""docstring for ClassName"""
	def __init__(self, graph):
		self.pq = list()
		self.dist = [np.inf] * graph.vertices
		self.graph = graph.wGraph
		self.parent = [-1] * graph.vertices
	
	def check(self):
		for nodes in self.graph.values():
			for node in nodes:
				if node[2] < 0:
					raise NameError("Dijkstra Algorithm cannot deal with negative weight ;(")

	def pushByHeap(self, node, newLen, descending=False):
		heappush(self.pq, [node, newLen])
		if descending is False:
			return nsmallest(len(self.pq), self.pq, key=lambda x: x[1])
		return nlargest(len(self.pq), self.pq, key=lambda x: x[1])

	def popByHeap(self, descending=False):
		if descending is False:
			nsmallest(len(self.pq), self.pq, key=lambda x: x[1])
			return heappop(self.pq)
		else:
			nlargest(len(self.pq), self.pq, key=lambda x: x[1])
			return heappop(self.pq)

	def dsp(self, start_node=0):
		self.dist[start_node] = 0
		
		for node in self.graph[start_node]:
			self.pq.append([node[1], node[2]])
			self.dist[node[1]] = node[2]
			self.parent[node[1]] = start_node

		while self.pq:
			u, uw = self.popByHeap()
			for node in self.graph[u]:
				_, v, vw = node
				newLen = self.dist[u] + vw
				if newLen < self.dist[v]:
					self.pushByHeap(v, newLen)
					self.dist[v] = newLen
					self.parent[v] = u
	def result(self):
		print("DSP: ", self.dist)



