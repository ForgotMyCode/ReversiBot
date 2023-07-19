import heuristic
import numpy as np

class CombinedHeuristic:
	"""
	Heuristic 'wrapper'. Wraps list of heuristic functions and provides functionality to combine them.
	"""

	def __init__(self, board_size=8):
		self.heuristic_functions = [
			heuristic.HEUR_corners,
			heuristic.HEUR_edges,
			heuristic.HEUR_mobility,
			heuristic.HEUR_parity,
			heuristic.HEUR_stability,
			heuristic.HEUR_corner_closeness,
			heuristic.HEUR_frontier,
			heuristic.HEUR_positional
		]

		edge_matrix_value = 100 / (2 * board_size + 2 * (board_size - 2))
		
		heuristic.EDGE_MATRIX = np.zeros((board_size, board_size))
		heuristic.EDGE_MATRIX[(0,-1), :] = edge_matrix_value
		heuristic.EDGE_MATRIX[:, (0,-1)] = edge_matrix_value

		heuristic.STABILITY_MATRIX = heuristic.STABILITY_MATRICES[board_size]
		heuristic.CORNER_CLOSENESS_MATRIX = heuristic.CORNER_CLOSENESS_MATRICES[board_size]
		heuristic.POSITIONAL_MATRIX = heuristic.POSITIONAL_MATRICES[board_size]

	def predict(self, data: heuristic.StatisticData):
		"""
		Returns heuristic evaluation of given data combined from every heuristic function.
		"""

		predictions = np.array([
			f(data) for f in self.heuristic_functions
		])
		
		better = np.sum(predictions > 0)
		worse = np.sum(predictions < 0)
		total = np.sum(predictions)

		return 10000*(better-worse) + total
