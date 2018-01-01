import os
import numpy as np
import pandas as pd
from collections import defaultdict


def main():
	metro = metroSys()
	metro.fit(time)

	while 1:
		source, destination, lessStop = ask()
		metro.init(source, destination, lessStop)
		metro.dsp()

class metroSys(object):
	"""docstring for metroSys"""
	def __init__(self):
		# Basic
		self.Graph = defaultdict(dict)
		self.V = 0
		# Prepare for algo
		self.costs = None
		self.parents = None
		self.path = None
		self.unvisited = None
		self.sourceList = None
		self.start = None
		self.destination = None
		self.prev = None
		self.total_cost = None
		self.lessStop = None

	def addEdge(self, source, destination, time):
		self.Graph[source][destination] = time

	def fit(self, Table):
		for i in range(len(Table)):
			source, destination, time = Table.iloc[i].tolist()
			self.addEdge(int(source), int(destination), float(time))

		self.V = len(Table)

	def init(self, source, destination, lessStop):
		# Initialize
		self.parents = defaultdict(dict)
		for key in self.Graph.keys():
			self.parents[key] = key
		self.costs = defaultdict(dict)
		for key in self.Graph.keys():
			self.costs[key]['t'] = float('inf')
			self.costs[key]['d'] = float('inf')
		self.unvisited = [True] * self.V
		self.path = list()
		self.prev = list()
		self.total_cost = 0
		self.lessStop = True if lessStop.lower() == "s" else False

		# For starting point
		source, destination = keysToValue[source], keysToValue[destination]
		# print(source, destination)
		self.start = source[0]
		self.destination = destination[0]
		self.sourceList = source
		self.destinationList = destination
		self.costs[self.start]['t'] = 0
		self.costs[self.start]['d'] = 0
		
	def dsp(self):
		# Setting
		start, destination, sourceList, destinationList = self.start, self.destination, self.sourceList, self.destinationList
		costs = self.costs
		parents = self.parents
		unvisited = self.unvisited
		prev = []; path = [];

		# Initialize
		node = start
		cost_time = costs[start]['t']
		cost_dist = costs[start]['d']
		neighbors = self.Graph[start]

		# Traverse
		while node != None:
			cost_time = costs[node]['t']
			cost_dist = costs[node]['d']
			neighbors = self.Graph[node]

			# Consider neighbors of neighbors
			candiate ={}
			tmp = [*neighbors]
			for ct in tmp:
				candiate.update({ct:neighbors[ct]})
			neighbors = {**neighbors, **candiate}
			# Prority pass for neighbor who is destination
			pq = False

			for neighbor in neighbors.keys():
				#if unvisited[neighbor] is True:
				# print("node: ", node, ",name: ",valueTokeys[node][0]);print("Set: ", neighbors);print("Considering: ", valueTokeys[neighbor][0], " (",neighbor,")")
				new_cost_time = cost_time + neighbors[neighbor]
				new_cost_dist = cost_dist + 1
				if costs[neighbor]['t'] > new_cost_time:
					costs[neighbor]['t'] = new_cost_time
					costs[neighbor]['d'] = new_cost_dist
					parents[neighbor] = node
					if not node in prev: prev.append(node)
				unvisited[node] = False

				if node in destinationList:
					p = node
					change = 0
					path.append(p)
					while not p in sourceList:
						prev = valueTokeys[p]
						p = parents[p]
						if prev == valueTokeys[p]:
							change += 1
						path.append(p)
					path.reverse()
					self.path = path
					self.total_cost = costs[node]['t']
					self.output_path(change)
					return 
				elif neighbor in destinationList:
					pq = neighbor

			node = self.popLowerCostNode(costs, unvisited, pq)

	def popLowerCostNode(self, costs, unvisited, pq=False):
		if pq != False:
			return pq
		lowest_cost_time = float("inf")
		lowest_cost_dist = float("inf")
		lowest_cost_node = None
		for node in costs:
			cost_time = costs[node]['t']
			cost_dist = costs[node]['d']
			if cost_time < lowest_cost_time and unvisited[node] is True:
				if cost_dist < lowest_cost_dist and self.lessStop is True:
					lowest_cost_time = cost_time
					lowest_cost_dist = cost_dist
					lowest_cost_node = node
				elif self.lessStop is False:
					lowest_cost_time = cost_time
					lowest_cost_node = node
		return lowest_cost_node


	def output_path(self, change):
		if self.start == self.destination:
			print("Cannot find a path between <{0}> and <{1}>".format(valueTokeys[self.start][0], valueTokeys[self.destination][0]))
			return None

		userChosen = "Quickest" if self.lessStop is False else "Shortest"
		print("The {0} Path from <{1}> to <{2}>:".format(userChosen, valueTokeys[self.start][0], valueTokeys[self.destination][0]))
		print(" ┌─ {0}".format(valueTokeys[self.start][0]))
		for station in self.path[1:len(self.path)-1]:
			if station != self.start or station != self.destination:
				print(" ├─ {0}".format(valueTokeys[station][0]))
		print(" └─ {0}".format(valueTokeys[self.destination][0]))
		print("The Result is Depended On <{0} Path>\nTotal stations: {1}\nChange Train: {2} Times\nTotal time: {3}".format(userChosen, len(self.path), change, self.total_cost))

# Simple Console UI

def ask():
	check = input("(R)oute Planning or (S)ee Stations List >> ").lower()
	if check == 'r':
		source = input("Which station are you from >> ")
		destination = input("Which station are you going >> ")
		if not source in keysToValue.keys():
			print("Check your <source> input, there might be some typo")
			return ask()
		elif not destination in keysToValue.keys():
			print("Check your <destination> input, there might be some typo")
			return ask()
		else:
			lessStop = input("You want to find a (Q)uickest path or a (S)hortest path >> ")
			if not lessStop.lower() in ["s", "q"]:
				print("Please insert an available option")
				return ask()
			return source, destination, lessStop
	elif check == 's':
		callStationNames()
		return ask()
	else:
		print("Please insert an available option")
		return ask()

def callStationNames():
	global names
	for i in range(len(names)):
		print("[{0}]\t{1}".format(i, names[i]))

# Data IO and Clean

def dataPrep():
	stationsCSV_path = os.path.join("data", "raw", "stations.csv")
	timeCSV_path = os.path.join("data", "raw", "time.csv")
	source_path = os.path.join("data", "txt", "metro_complet.txt")

	if not os.path.isfile(stationsCSV_path) or not os.path.isfile(timeCSV_path):
		dataClean()
		stations = pd.read_csv(stationsCSV_path, delimiter=";", skipinitialspace=True)
		time = pd.read_csv(timeCSV_path, delimiter=";", skipinitialspace=True)
	else:
		stations = pd.read_csv(stationsCSV_path, delimiter=";", skipinitialspace=True)
		time = pd.read_csv(timeCSV_path, delimiter=";", skipinitialspace=True)

	keysToValue = defaultdict(list)
	for i in range(len(stations)):
		id_, station = stations.iloc[i].tolist()
		keysToValue[station].append(id_)

	valueTokeys = defaultdict(list)
	for i in range(len(stations)):
		id_, station = stations.iloc[i].tolist()
		valueTokeys[id_].append(station)

	names = sorted(list(set(stations.station.tolist())))

	return stations, time, keysToValue, valueTokeys, names

def dataClean():
	stationsCSV_path = os.path.join("data", "raw", "stations.csv")
	timeCSV_path = os.path.join("data", "raw", "time.csv")
	source_path = os.path.join("data", "txt", "metro_complet.txt")

	with open(source_path, "r", encoding="ISO-8859-15") as file:
		metro = file.read()

	vertices, edges = metro.split("[Edges]\n")[0], metro.split("[Edges]\n")[1]
	vertices = vertices.split("\n")[1:377]
	edges = edges.split("\n")[0:933]

	with open(stationsCSV_path, "w") as file:
		head = "id" + ";" + "station" + "\n"
		file.write(head)
		for station in vertices:
			line = station[:4] + ";" + station[5:] + "\n"
			file.write(line)

	with open(timeCSV_path, "w") as file:
		head = "source" + ";" + "destination" + ";" + "time" + "\n"
		file.write(head)
		for edge in edges:
			source, destination, time = edge.split(" ")
			line = source + ";" + destination + ";" + time + "\n"
			file.write(line)

stations, time, keysToValue, valueTokeys, names = dataPrep()

'''
metro = metroSys()
metro.fit(time)

while 1:
	source = input("Which station are you from >> ")
	destination = input("Which station are you going >> ")
	metro.init(source, destination)
	metro.dsp()'''


if __name__ == '__main__':
	main()
