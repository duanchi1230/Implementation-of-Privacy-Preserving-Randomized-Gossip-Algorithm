import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

graph_size = 100
k = 2.5

def generate_bipartite(graph_size=100, left_to_total_size_ratio=2):
	"""
	This function generates the fully connected bipartite graph.
	:param graph_size: The total nodes in the bipartite graph.
	:param left_to_total_size_ratio: The ration of the total nodes to the left nodes.
	:return: node vector of size n, and edge matrix of size n*n with entries of e_ij = 1 for i,j belongs
			 to the edge of the graph.
	"""
	k = left_to_total_size_ratio
	# Generate the edge for bipartite graph
	edge = pd.DataFrame(np.zeros((graph_size, graph_size)))
	for i in range(int(graph_size / k)):
		edge.iloc[i, range(int(graph_size / k), graph_size)] = 1
	for i in range(int(graph_size / k), graph_size):
		edge.iloc[i, range(int(graph_size / k))] = 1
	# Generate values for each node by random from uniform distribution [0,1]
	node = np.random.uniform(size=graph_size)
	return node, edge


def generate_inverse_bipartite(graph_size=100, left_to_total_size_ratio=2):
	"""
	This function generates an sparse connected inverse complete bipartite graph
	:param graph_size: The total nodes in the bipartite graph
	:param left_to_total_size_ratio: The ration of the total nodes to the left nodes.
	:return: node vector of size n, and edge matrix of size n*n with entries of e_ij = 1 for i,j belongs
			 to the edge of the graph.
	"""
	k = left_to_total_size_ratio
	# Generate the edge for inverse bipartite graph
	edge = pd.DataFrame(np.zeros((graph_size, graph_size)))
	for i in range(int(graph_size / k)):
		edge.iloc[i, range(i, int(graph_size / k))] = 1
	for i in range(int(graph_size / k), graph_size):
		edge.iloc[i, range(int(graph_size / k), graph_size)] = 1
	# Generate values for each node by random from uniform distribution [0,1]
	node = np.random.uniform(size=graph_size)
	# Connect left node to right by one edge
	for i in range(len(edge)):
		edge.iloc[i][i] = 0
	edge.iloc[0][len(edge) - 1] = 1
	return node, edge

# node, edge = generate_bipartite(graph_size=100, left_to_total_size_ratio=k)
# print(edge)

node, edge = generate_inverse_bipartite(graph_size=100, left_to_total_size_ratio=k)
print(edge)

def calculate_relative_error(x_t, x_0):
	"""
	Calculate the relative error according to the measure:
		error=||x_t-x_star||^2/||x_0-x_star||^2, where
	x_star=sum(x_0_i)/n for i in x_0, and n is the total of elements in x_0
	:param x_t: The node value vector at step t
	:param x_0: The intial node value
	:return: Relative error (scalar)
	"""
	x_star = np.full(len(x_0), x_0.sum() / len(x_0))
	relative_error = np.square(x_t - x_star).sum() / np.square(x_0 - x_star).sum()
	return relative_error

def calculate_adaptive_rate(node, e_matrix):
	"""
	Calculate the aaptive rate for the Binary Oracle algorithm as following:
		adaptive_rate=1/(4*m) * sum(x_t_i, x_t_j) for all i,j belonging to the edge(i,j)
	:param node: The node valus at step t
	:param e_matrix: matrix of size m*2, containing all edges indexes i,j in each row
	:return: adaptive rate (scalar)
	"""
	adaptive_rate = 0
	for e_ in e_matrix:
		adaptive_rate = adaptive_rate + abs(node[e_[0]] - node[e_[1]])
	adaptive_rate = adaptive_rate / (4 * len(e_matrix))
	return adaptive_rate

def Pairwise_Gossip(node, edge, iterations=3000):
	"""
	This moule is the implementation of the pairwise gossip algorithm for averaging consensus problem
	:param node: Intial node values vector for the graph
	:param edge: Edge matrix of size n*n with entries of e_ij = 1 for i,j belongs
			     to the edge of the graph.
	:param iterations: Total number of iterations of the algorithm run
	:return: x_t: the node value at the end of the algorithm
			 relative error: Vector of the relative error at each step
			 node_value_at_each_step: Matrix of size t*n, whose rows are vectors of
			                          node values at each step
	"""
	# Generate matrix of size m*2, containing all edges indexes i,j in each row
	matrix_edge = []
	for i in range(len(edge)):
		for j in range(len(edge.iloc[0])):
			if edge.iloc[i][j] == 1:
				matrix_edge.append([i, j])
	# Draw the random edge for each iteration
	random_edge = np.random.randint(len(matrix_edge), size=iterations)
	# Initialize node value for first step
	x_t = np.copy(node)
	node_value_at_each_step = []
	relative_error = []
	# Run algorithm
	for e in random_edge:
		x_average = (x_t[matrix_edge[e][0]] + x_t[matrix_edge[e][1]]) / 2
		x_t[matrix_edge[e][0]] = x_average
		x_t[matrix_edge[e][1]] = x_average
		node_value_at_each_step.append(np.copy(x_t))
		relative_error.append(calculate_relative_error(x_t, node))

	return x_t, relative_error, node_value_at_each_step

def Binary_Oracle(node, edge, iterations=3000, l_rate=0.01):
	"""
	This moule is the implementation of the Binary-Oricale privacy preserving algorithm
	for averaging consensus problem
	:param node: Intial node values vector for the graph
	:param edge: Edge matrix of size n*n with entries of e_ij = 1 for i,j belongs
			     to the edge of the graph.
	:param iterations: Total number of iterations of the algorithm run
	:param l_rate: The learning lambda value with options:
					1. numeric value
					2. string: 'inverse' for l_rat=1/t, 'sqrt_invese' for l_rat=1/sqrt(t), or
								'adaptive' for l_rate=1/(4*m) * sum(x_t_i, x_t_j)
								for all i,j belonging to the edge(i,j)

	:return: x_t: the node value at the end of the algorithm
			 relative error: Vector of the relative error at each step
			 node_value_at_each_step: Matrix of size t*n, whose rows are vectors of
			                          node values at each step
	"""
	# Generate matrix of size m*2, containing all edges indexes i,j in each row
	matrix_edge = []
	for i in range(len(edge)):
		for j in range(len(edge.iloc[0])):
			if edge.iloc[i][j] == 1:
				matrix_edge.append([i, j])
	# Draw the random edge for each iteration
	random_edge = np.random.randint(len(matrix_edge), size=iterations)
	# Initialize node value for first step
	x_t = np.copy(node)
	node_value_at_each_step = []
	relative_error = []
	rate_option = {'inverse': '1/(i+1)', 'sqrt_inverse': '1/np.sqrt(i+1)',
	               'adaptive': 'calculate_adaptive_rate(x_t, matrix_edge)'}
	# Run algorithm
	# When input l_rate is numeric
	if type(l_rate) != type('str'):
		for i in range(len(random_edge)):
			if x_t[matrix_edge[random_edge[i]][0]] >= x_t[matrix_edge[random_edge[i]][1]]:
				x_t[matrix_edge[random_edge[i]][0]] = x_t[matrix_edge[random_edge[i]][0]] - l_rate
				x_t[matrix_edge[random_edge[i]][1]] = x_t[matrix_edge[random_edge[i]][1]] + l_rate
			if x_t[matrix_edge[random_edge[i]][0]] < x_t[matrix_edge[random_edge[i]][1]]:
				x_t[matrix_edge[random_edge[i]][0]] = x_t[matrix_edge[random_edge[i]][0]] + l_rate
				x_t[matrix_edge[random_edge[i]][1]] = x_t[matrix_edge[random_edge[i]][1]] - l_rate
			node_value_at_each_step.append(np.copy(x_t))
			relative_error.append(calculate_relative_error(x_t, node))
	# When input l_rate is string type
	if type(l_rate) == type('str'):
		for i in range(len(random_edge)):
			if x_t[matrix_edge[random_edge[i]][0]] >= x_t[matrix_edge[random_edge[i]][1]]:
				x_t[matrix_edge[random_edge[i]][0]] = x_t[matrix_edge[random_edge[i]][0]] - eval(rate_option[l_rate])
				x_t[matrix_edge[random_edge[i]][1]] = x_t[matrix_edge[random_edge[i]][1]] + eval(rate_option[l_rate])

			if x_t[matrix_edge[random_edge[i]][0]] < x_t[matrix_edge[random_edge[i]][1]]:
				x_t[matrix_edge[random_edge[i]][0]] = x_t[matrix_edge[random_edge[i]][0]] + eval(rate_option[l_rate])
				x_t[matrix_edge[random_edge[i]][1]] = x_t[matrix_edge[random_edge[i]][1]] - eval(rate_option[l_rate])
			node_value_at_each_step.append(np.copy(x_t))
			relative_error.append(calculate_relative_error(x_t, node))
			print(i)
	print(relative_error)
	print(x_t)
	return x_t, relative_error, node_value_at_each_step

def e_Gap_Oracle(node, edge, iterations=100000, e_rate=0.02):
	"""
	This moule is the implementation of the e-Gap-Oricale privacy preserving algorithm
	for averaging consensus problem
	:param node: Intial node values vector for the graph
	:param edge: Edge matrix of size n*n with entries of e_ij = 1 for i,j belongs
			     to the edge of the graph.
	:param iterations: Total number of iterations of the algorithm run
	:param e_rate: The learning e-Gap value with numeric values
	:return: x_t: the node value at the end of the algorithm
			 relative error: Vector of the relative error at each step
			 node_value_at_each_step: Matrix of size t*n, whose rows are vectors of
			                          node values at each step
	"""
	# Generate matrix of size m*2, containing all edges indexes i,j in each row
	matrix_edge = []
	for i in range(len(edge)):
		for j in range(len(edge.iloc[0])):
			if edge.iloc[i][j] == 1:
				matrix_edge.append([i, j])
	# Draw the random edge for each iteration
	random_edge = np.random.randint(len(matrix_edge), size=iterations)
	# Initialize node value for first step
	x_t = np.copy(node)
	node_value_at_each_step = []
	relative_error = []
	# Run algorithm
	for e in random_edge:
		if x_t[matrix_edge[e][0]] - x_t[matrix_edge[e][1]] >= e_rate:
			x_t[matrix_edge[e][0]] = x_t[matrix_edge[e][0]] - e_rate / 2
			x_t[matrix_edge[e][1]] = x_t[matrix_edge[e][1]] + e_rate / 2
		if x_t[matrix_edge[e][0]] - x_t[matrix_edge[e][1]] <= -e_rate:
			x_t[matrix_edge[e][0]] = x_t[matrix_edge[e][0]] + e_rate / 2
			x_t[matrix_edge[e][1]] = x_t[matrix_edge[e][1]] - e_rate / 2
		node_value_at_each_step.append(np.copy(x_t))
		relative_error.append(calculate_relative_error(x_t, node))
	print(relative_error)
	print(x_t)
	return x_t, relative_error, node_value_at_each_step

def Controlled_Noise_Insertion(node, edge, iterations=1000, decay_rate=0.995):
	"""
	This moule is the implementation of the e-Gap-Oricale privacy preserving algorithm
	for averaging consensus problem
	:param node: Intial node values vector for the graph
	:param edge: Edge matrix of size n*n with entries of e_ij = 1 for i,j belongs
			     to the edge of the graph.
	:param iterations: Total number of iterations of the algorithm run
	:param decay_rate: The decay rate with numeric values for the inserted normally distributed
	                    noise variance
	:return: x_t: the node value at the end of the algorithm
			 relative error: Vector of the relative error at each step
			 node_value_at_each_step: Matrix of size t*n, whose rows are vectors of
			                          node values at each step
	"""
	# Generate matrix of size m*2, containing all edges indexes i,j in each row
	matrix_edge = []
	for i in range(len(edge)):
		for j in range(len(edge.iloc[0])):
			if edge.iloc[i][j] == 1:
				matrix_edge.append([i, j])
	# Draw the random edge for each iteration
	random_edge = np.random.randint(len(matrix_edge), size=iterations)
	# Initialize node value for first step
	x_t = np.copy(node)
	node_value_at_each_step = []
	relative_error = []
	v_i = 0
	v_j = 0
	k = 1
	# Run algorithm
	for e in random_edge:
		vi_t = np.random.normal(loc=0, scale=1)
		vj_t = np.random.normal(loc=0, scale=1)
		w_i = decay_rate ** k * vi_t - decay_rate ** (k - 1) * v_i
		w_j = decay_rate ** k * vj_t - decay_rate ** (k - 1) * v_j
		x_average = (x_t[matrix_edge[e][0]] + w_i + x_t[matrix_edge[e][1]] + w_j) / 2
		x_t[matrix_edge[e][0]] = x_average
		x_t[matrix_edge[e][1]] = x_average
		node_value_at_each_step.append(np.copy(x_t))
		relative_error.append(calculate_relative_error(x_t, node))
		v_i = vi_t
		v_j = vj_t
		k = k + 1
	print(relative_error)
	print(x_t)
	return x_t, relative_error, node_value_at_each_step

# Alternating the l_rate will give the convergence results for Binary Oracle algorithm
iterations = 5000
l_rate = 0.001
x_t, relative_error, node_value_at_each_step = Binary_Oracle(node,edge, iterations=iterations, l_rate=l_rate)
node_value_at_each_step = pd.DataFrame(node_value_at_each_step).transpose()
print(node_value_at_each_step)
for i in range(len(node_value_at_each_step)):
	plt.plot(range(iterations), node_value_at_each_step.iloc[i])
# l_rate = '1/i'
plt.xlabel('Iterations \n Lambda='+str(l_rate))
plt.ylabel('Node value at each step')
plt.title('Convergence for inverse bipartite graph with lambda='+ str(l_rate))
plt.show()

# Alternating the e_rate will give the convergence results for e-Gap Oracle algorithm
e_rate = 0.001
x_t, relative_error, node_value_at_each_step = e_Gap_Oracle(node,edge, iterations=iterations, e_rate=e_rate)
node_value_at_each_step = pd.DataFrame(node_value_at_each_step).transpose()
print(node_value_at_each_step)
for i in range(len(node_value_at_each_step)):
	plt.plot(range(iterations), node_value_at_each_step.iloc[i])
plt.xlabel('Iterations \n Lambda='+str(l_rate))
plt.ylabel('Node value at each step')
plt.title('Convergence for inverse bipartite graph with lambda='+ str(l_rate))
plt.show()

# Alternating the decay_rate will give the convergence results for Controlled noise Insertion algorithm
decay_rate = 0.9995
x_t, relative_error, node_value_at_each_step = Controlled_Noise_Insertion(node,edge, iterations=iterations, decay_rate=decay_rate)
node_value_at_each_step = pd.DataFrame(node_value_at_each_step).transpose()
print(node_value_at_each_step)
for i in range(len(node_value_at_each_step)):
	plt.plot(range(iterations), node_value_at_each_step.iloc[i])
plt.xlabel('Iterations \n decay_rate=' + str(decay_rate))
plt.ylabel('Node value at each step')
plt.title('Convergence for inverse bipartite graph with decay_rate='+str(decay_rate))
plt.show()

# Alternating the algorithms will give the relative error plots
# Rates for Binary Oracle algorithm
# rate = [0.001, 0.01, 0.1, 'inverse', 'sqrt_inverse', 'adaptive']
# name = [0.001, 0.01, 0.1, '1/i', '1/sqrt(i)', 'adaptive', 'Baseline']
# Rates for e-Gap Oracle algorithm
# rate = [0.002, 0.02,0.2]
# name = [0.002, 0.02,0.2]
# Rates for Controlled Noise Insertion algorithm
rate = [0.001, 0.01, 0.1, 0.5, 0.995, 0.9995]
name = [0.001, 0.01, 0.1, 0.5, 0.995, 0.9995, 'Baseline']
for i in range(len(name)):
	name[i] = 'decay_rate=' + str(name[i])
name[-1] = 'Baseline'
print(name)
error = []
for l in rate:
	x_t, relative_error, node_value_at_each_step = Controlled_Noise_Insertion(node, edge, iterations=iterations,
	                                                                          decay_rate=l)
	error.append(relative_error)
x_t, relative_error, node_value_at_each_step = Pairwise_Gossip(node, edge, iterations=iterations)
error.append(relative_error)
for i in range(len(error)):
	plt.plot(range(iterations), error[i], label=str(name[i]))
plt.legend(loc=0)
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Relative Error")
plt.title('Controlled Noise Insertion-Log Scale')
plt.show()