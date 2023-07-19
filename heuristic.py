import numpy as np

class StatisticData:
	"""
	Simple structure that holds data that are interesting for heuristic functions.
	"""
	def __init__(self,
		my_move_count,
		opponent_move_count,

		shallow_board,

		score,

		scale = 100.0
		):

		self.scale = scale
		self.my_move_count = my_move_count
		self.opponent_move_count = opponent_move_count
		self.mat = shallow_board

		self.score = score

	def simple_from_board(board):
		my_move_count = len(board.get_all_valid_moves(1) or [])
		opponent_move_count = len(board.get_all_valid_moves(-1) or [])
		shallow_board = np.array(board.board)
		score = np.sum(shallow_board)
		return StatisticData(my_move_count, opponent_move_count, shallow_board, score)

	def from_board_and_colors(raw_board, shallow_board, my_color, opponent_color):
		my_move_count = len(raw_board.get_all_valid_moves(my_color) or [])
		opponent_move_count = len(raw_board.get_all_valid_moves(opponent_color) or [])
		score = np.sum(shallow_board)
		return StatisticData(my_move_count, opponent_move_count, shallow_board, score)

# The following are implemented heuristic functions:
# Some were inspired by https://barberalec.github.io/pdf/An_Analysis_of_Othello_AI_Strategies.pdf

def HEUR_parity(data: StatisticData) -> float:
	"""
	Heuristic function that takes into account the difference of fields
	captured by max - min player.
	
	Returns evaluation, in most cases in range -100 - 100.
	"""

	scale = 100 / (data.mat.shape[0] * data.mat.shape[1])

	moves = np.sum(np.abs(data.mat))

	if moves <= data.mat.shape[0]:
		# it is better to give away stones early
		return -scale * data.score
	#else
	return scale * data.score

STABILITY_MATRIX = None

STABILITY_MATRICES = {
	8:

# inspired by https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf
np.array(
	[
		[4,-3,2,2,2,2,-3,4],
		[-3,-4,-1,-1,-1,-1,-4,-3],
		[2,-1,1,0,0,1,-1,2],
		[2,-1,0,1,1,0,-1,2],
		[2,-1,0,1,1,0,-1,2],
		[2,-1,1,0,0,1,-1,2],
		[-3,-4,-1,-1,-1,-1,-4,-3],
		[4,-3,2,2,2,2,-3,4],
	]
),

10: np.array(
	[
		[4,-3,2,2,2,2,2,2,-3,4],
		[-3,-4,-1,-1,-1,-1,-1,-1,-4,-3],
		[2,-1,1,0,0,0,0,1,-1,2],
		[2,-1,0,0,0,0,0,0,-1,2],
		[2,-1,0,0,1,1,0,0,-1,2],
		[2,-1,0,0,1,1,0,0,-1,2],
		[2,-1,0,0,0,0,0,0,-1,2],
		[2,-1,1,0,0,0,0,1,-1,2],
		[-3,-4,-1,-1,-1,-1,-1,-1,-4,-3],
		[4,-3,2,2,2,2,2,2,-3,4],
	]
),

6: np.array(
	[
		[4,-3,2,2,-3,4],
		[-3,-4,-1,-1,-4,-3],
		[2,-1,1,1,-1,2],
		[2,-1,1,1,-1,2],
		[-3,-4,-1,-1,-4,-3],
		[4,-3,2,2,-3,4],
	]
)
}

def HEUR_stability(data: StatisticData) -> float:
	"""
	Heuristic function that takes into account the difference of stability
	of fields.
	
	Returns evaluation, in most cases in range -100 - 100.
	"""

	stability = np.sum(data.mat * STABILITY_MATRIX)

	return stability

def HEUR_corners(data: StatisticData) -> float:
	"""
	Heuristic function that takes into account the difference in
	counts of corners captured.
	
	Returns evaluation, in most cases in range -100 - 100.
	"""

	return 25 * (data.mat[0,0] + data.mat[0, -1] + data.mat[-1, 0] + data.mat[-1, -1])

EDGE_MATRIX = None
#np.array(
#	[
#		[4, 4, 4, 4, 4, 4, 4, 4],
#		[4, 0, 0, 0, 0, 0, 0, 4],
#		[4, 0, 0, 0, 0, 0, 0, 4],
#		[4, 0, 0, 0, 0, 0, 0, 4],
#		[4, 0, 0, 0, 0, 0, 0, 4],
#		[4, 0, 0, 0, 0, 0, 0, 4],
#		[4, 0, 0, 0, 0, 0, 0, 4],
#		[4, 4, 4, 4, 4, 4, 4, 4],
#	]
#)

def HEUR_edges(data: StatisticData) -> float:
	"""
	Heuristic function that takes into account the difference of captures
	on edges.
	
	Returns evaluation, in most cases in range -100 - 100.
	"""

	edges = np.sum(data.mat * EDGE_MATRIX)

	return edges

def HEUR_mobility(data: StatisticData) -> float:
	"""
	Heuristic function that takes into account the difference of mobility
	of players.
	
	Returns evaluation  range -scale - scale.
	"""

	total = data.my_move_count + data.opponent_move_count
	diff = data.my_move_count - data.opponent_move_count

	if total == 0:
		return 0

	return data.scale * diff / total

CORNER_CLOSENESS_MATRIX = None

CORNER_CLOSENESS_MATRICES = {
	8:
	np.array([
		[27, -9, 0, 0, 0, 0, -9, 27],
		[-9, -9, 0, 0, 0, 0, -9, -9],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[-9, -9, 0, 0, 0, 0, -9, -9],
		[27, -9, 0, 0, 0, 0, -9, 27],
	]),

	10:
	np.array([
		[27, -9, 0, 0, 0, 0, 0, 0, -9, 27],
		[-9, -9, 0, 0, 0, 0, 0, 0, -9, -9],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[-9, -9, 0, 0, 0, 0, 0, 0, -9, -9],
		[27, -9, 0, 0, 0, 0, 0, 0, -9, 27],
	]),

	6:
	np.array([
		[27, -9, 0, 0, -9, 27],
		[-9, -9, 0, 0, -9, -9],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[-9, -9, 0, 0, -9, -9],
		[27, -9, 0, 0, -9, 27],
	]),
}

def HEUR_corner_closeness(data: StatisticData) -> float:
	"""
	Heuristic function that takes into account the difference of captured
	fields adjacent to corners.
	
	Returns evaluation, in most cases in range -100 - 100.
	"""

	corner_closeness = np.sum(data.mat * CORNER_CLOSENESS_MATRIX)

	return corner_closeness

def HEUR_frontier(data: StatisticData) -> float:
	"""
	Heuristic function that takes into account the difference of disks
	adjacent to empty space(s).
	
	Returns evaluation, in most cases in range -100 - 100.
	"""

	mask = np.zeros(data.mat.shape)

	value = 100 / ((data.mat.shape[0] - 1) * (data.mat.shape[1] - 1))
	half_value = value / 2

	# handle inner cells
	for y in range(1, data.mat.shape[1]):
		for x in range(1, data.mat.shape[0]):
			if data.mat[y,x] != 0:
				# if there are no empty squares, product is non-zero, 0 is written
				# if there are empty squares, product is zero, -x is written
				mask[y,x] = -value * int(not np.prod(data.mat[y-1:y+2, x-1:x+2]))

	# handle y edges
	for y in range(1, data.mat.shape[1]):
		if data.mat[y,0] != 0:
			mask[y,0] = -half_value * int(not np.prod(data.mat[y-1:y+2, 0:2]))

		if data.mat[y,-1] != 0:
			mask[y,-1] = -half_value * int(not np.prod(data.mat[y-1:y+2, -2:]))

	# handle x edges
	for x in range(1, data.mat.shape[1]):
		if data.mat[0,x] != 0:
			mask[0,x] = -half_value * int(not np.prod(data.mat[0:2, x-1:x+2]))

		if data.mat[-1,x] != 0:
			mask[-1,x] = -half_value * int(not np.prod(data.mat[-2:, x-1:x+2]))

	# ignore corners

	frontier = np.sum(data.mat * mask)

	return frontier

POSITIONAL_MATRIX = None

POSITIONAL_MATRICES = {
	8:

# inspired by https://www.samsoft.org.uk/reversi/strategy.htm#stable
np.array(
	[
		[99,-8,8,6,6,8,-8,99],
		[-8,-24,-4,-3,-3,-4,-24,-8],
		[8,-4,7,4,4,7,-4,8],
		[6,-3,4,0,0,4,-3,6],
		[6,-3,4,0,0,4,-3,6],
		[8,-4,7,4,4,7,-4,8],
		[-8,-24,-4,-3,-3,-4,-24,-8],
		[99,-8,8,6,6,8,-8,99],
	]
),

10:
np.array(
	[
		[99,-8,8,6,6,6,6,8,-8,99],
		[-8,-24,-4,-3,-3,-3,-3,-4,-24,-8],
		[8,-4,7,4,4,4,4,7,-4,8],
		[6,-3,4,0,0,0,0,4,-3,6],
		[6,-3,4,0,0,0,0,4,-3,6],
		[6,-3,4,0,0,0,0,4,-3,6],
		[6,-3,4,0,0,0,0,4,-3,6],
		[8,-4,7,4,4,4,4,7,-4,8],
		[-8,-24,-4,-3,-3,-3,-3,-4,-24,-8],
		[99,-8,8,6,6,6,6,8,-8,99],
	]
),

6:
np.array(
	[
		[99,-8,8,8,-8,99],
		[-8,-24,-4,-4,-24,-8],
		[8,-4,0,0,-4,8],
		[8,-4,0,0,-4,8],
		[-8,-24,-4,-4,-24,-8],
		[99,-8,8,8,-8,99],
	]
),
}

def HEUR_positional(data: StatisticData) -> float:
	positional_score = np.sum(data.mat * POSITIONAL_MATRIX)

	return positional_score	
