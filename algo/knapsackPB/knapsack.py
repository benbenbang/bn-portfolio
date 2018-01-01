from itertools import groupby

maxwt = int(input("Insert a number for the maximum weight of the sack >> "))
 
'''stores = (
			("map", 9, 150, 1),
			("compass", 13, 35, 1),
			("water", 153, 200, 3),
			("sandwich", 50, 60, 2),
			("glucose", 15, 60, 2),
			("tin", 68, 45, 3),
			("banana", 27, 60, 3),
			("apple", 39, 40, 3),
			("cheese", 23, 30, 1),
			("beer", 52, 10, 3),
			("suntan cream", 11, 70, 1),
			("camera", 32, 30, 1),
			("t-shirt", 24, 15, 2),
			("trousers", 48, 10, 2),
			("umbrella", 73, 40, 1),
			("waterproof trousers", 42, 70, 1),
			("waterproof overclothes", 43, 75, 1),
			("breifcase", 22, 80, 1),
			("sunglasses", 7, 20, 1),
			("towel", 18, 12, 2),
			("socks", 4, 50, 1),
			("book", 30, 10, 2)
		)'''


stores = (
		("x1", 4, 8, 1),
		("x2", 3, 9, 1),
		("x3", 2, 6, 1)
	)

items = sum( ([(item, wt, val)] * n for item, wt, val, n in stores), [])
 
def knapsack01(items, limit):
	table = [[0 for w in range(limit + 1)] for j in range(len(items) + 1)]
 
	for j in range(1, len(items) + 1):
		item, wt, val = items[j-1]
		for w in range(1, limit + 1):
			if wt > w:
				table[j][w] = table[j-1][w]
			else:
				table[j][w] = max(table[j-1][w],
								  table[j-1][w-wt] + val)
 
	result = []
	w = limit
	for j in range(len(items), 0, -1):
		was_added = table[j][w] != table[j-1][w]
 
		if was_added:
			item, wt, val = items[j-1]
			result.append(items[j-1])
			w -= wt
 
	return result
 
 
bagged = knapsack01(items, maxwt)
print("Bagged the following %i items\n  " % len(bagged) +
	  '\n  '.join('%i off: %s' % (len(list(grp)), item[0])
				  for item,grp in groupby(sorted(bagged))))
print("for a total value of %i and a total weight of %i" % (
	sum(item[2] for item in bagged), sum(item[1] for item in bagged)))