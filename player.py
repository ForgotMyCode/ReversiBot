from transpos_table import TranspositionTable
import math
import copy
import numpy as np
from combined_heuristic_combo import CombinedHeuristic
from heuristic import StatisticData
import time

class MyPlayer():
	'''Iterative Deepening Negamax, A-B pruning, LookUp Tables + 8 Combined Heuristics'''

	TIME_LIMIT = 5.0
	INITIAL_TIME_DOTATION = 59
	START_DEPTH = 1

	def __init__(self, my_color,opponent_color, board_size=8):
		self.name = 'mezlondr'
		self.my_color = my_color
		self.opponent_color = opponent_color
		self.board_size = board_size
		self.transposition_table = TranspositionTable()
		self.heuristic = CombinedHeuristic(board_size)
		self.critical_time = math.inf

		# warm-up, and cache some values´
		warm_up_board1 = [[-1 for _ in range(board_size)] for _ in range(board_size)]
		warm_up_board2 = [[-1 for _ in range(board_size)] for _ in range(board_size)]

		idx1 = board_size // 2
		idx2 = idx1 - 1
		warm_up_board1[idx1][idx1] = self.my_color
		warm_up_board1[idx1][idx2] = self.opponent_color
		warm_up_board1[idx2][idx1] = self.opponent_color
		warm_up_board1[idx2][idx2] = self.my_color

		warm_up_board2[idx1][idx1] = self.my_color
		warm_up_board2[idx1][idx2] = self.opponent_color
		warm_up_board2[idx2][idx1] = self.opponent_color
		warm_up_board2[idx2][idx2] = self.my_color

		# utilizing 60 seconds for construction as 2x 30 seconds
		_time_limit = MyPlayer.TIME_LIMIT
		MyPlayer.TIME_LIMIT = (MyPlayer.INITIAL_TIME_DOTATION-0.5)/2

		self.move(warm_up_board1)
		self.move(warm_up_board2)

		# set the actual time limit
		MyPlayer.TIME_LIMIT = _time_limit

	def heuristic_evaluate(self, current_move_count, opponent_move_count, board_copy) -> float:
		"""
		Quickly evaluate inner negamax state using heuristics.
		"""

		data = StatisticData(current_move_count, opponent_move_count, board_copy, self.__get_score(board_copy))
		evaluation = self.heuristic.predict(data)
		return evaluation

	def negamax_evaluate(self, board_copy, alpha, beta, depth, opponent_move_count = 4) -> float:
		"""
		Negamax evaluation of current game state.
		"""

		# inspired by: https://en.wikipedia.org/wiki/Negamax

		# print(depth)
		alpha_orig = alpha

		key = TranspositionTable.get_key(board_copy)

		tt_entry = self.transposition_table.lookup(key)

		if tt_entry is not None and tt_entry.depth >= depth:
			if tt_entry.flag == TranspositionTable.FLAG_EXACT:
				return tt_entry.value
			#else
			if tt_entry.flag == TranspositionTable.FLAG_LOWERBOUND:
				alpha = max(alpha, tt_entry.value)
			else:
				# flag == UPPERBOUND
				beta = min(beta, tt_entry.value)

			if alpha >= beta:
				return tt_entry.value

		valid_moves = self.get_all_valid_moves(board_copy)
		current_move_count = len(valid_moves)

		if current_move_count == 0:
			score = self.__get_score(board_copy)
			if score == 0:
				return 0
			return 1000000 * score / abs(score)

		if depth == 0:
			return self.heuristic_evaluate(current_move_count, opponent_move_count, board_copy)

		value = -math.inf

		for move in valid_moves:
			if time.time() >= self.critical_time:				
				return self.heuristic_evaluate(current_move_count, opponent_move_count, board_copy)

			child = board_copy.copy()
			self.__play_move(move, child)

			value = max(value, -self.negamax_evaluate(-child, -beta, -alpha, depth-1, current_move_count))

			alpha = max(alpha, value)

			if alpha >= beta:
				break
		
		tt_entry = TranspositionTable.TranspositionTableEntry(value, depth, TranspositionTable.FLAG_EXACT)

		if value <= alpha_orig:
			tt_entry.flag = TranspositionTable.FLAG_UPPERBOUND
		elif value >= beta:
			tt_entry.flag = TranspositionTable.FLAG_LOWERBOUND
		
		self.transposition_table.store(key, tt_entry)

		return value

	def move(self,board):
		"""
		Make a move.
		"""

		self.critical_time = time.time() + MyPlayer.TIME_LIMIT - 0.1

		mat = MyPlayer.convert_board(board, self.board_size, self.my_color, self.opponent_color)
		
		valid_moves = self.get_all_valid_moves(mat)

		if len(valid_moves) == 0:
			return None
		
		# iterative deepening

		best_arg = -1
		best_value = math.inf

		max_depth = MyPlayer.START_DEPTH

		while True:
			max_depth += 1
			_best_arg = -1
			_best_value = math.inf

			for i, move in enumerate(valid_moves):				
				if time.time() >= self.critical_time:
					break

				child = mat.copy()
				self.__play_move(move, child)

				evaluation = self.negamax_evaluate(-child, -math.inf, math.inf, max_depth)

				if evaluation < _best_value:
					_best_value = evaluation
					_best_arg = i

			else:
				best_value = _best_value
				best_arg = _best_arg
				continue
			break # inner loop break will break the outer one as well

		
		# print(max_depth)
		# print("Evaluation for opponent:", best_value)
		# print(time.time() - self.critical_time)
		
		return valid_moves[best_arg]

	def convert_board(board, board_size, my_color, opponent_color):
		"""
		Converts a standard board into my representation.
		"""

		# I am using different board representation, since it is more suitable for the calculations I am doing.
		# The board is from now on a numpy array, where 1 is current player's stone, -1 is opponent player's stone
		# and 0 is an empty space.

		mat = np.zeros((board_size, board_size))

		for i in range(board_size):
			for j in range(board_size):
				if board[i][j] == my_color:
					mat[i, j] = 1
				elif board[i][j] == opponent_color:
					mat[i,j] = -1
		
		return mat

# The methods below are pretty much stolen from the files given, or already existed in the template.
# Only very slight modifications were made.

	def __is_correct_move(self,move,board):
		if board[move[0],move[1]] == 0:
			dx = [-1,-1,-1,0,1,1,1,0]
			dy = [-1,0,1,1,1,0,-1,-1]
			for i in range(len(dx)):
				if self.__confirm_direction(move,dx[i],dy[i],board):
					return True

		return False

	def __confirm_direction(self,move,dx,dy,board):
		posx = move[0]+dx
		posy = move[1]+dy
		if (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
			if board[posx,posy] == -1:
				while (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
					posx += dx
					posy += dy
					if (posx>=0) and (posx<self.board_size) and (posy>=0) and (posy<self.board_size):
						if board[posx,posy] == 0:
							return False
						if board[posx,posy] == 1:
							return True

		return False

	def __get_score(self, board):
		return np.sum(board)

	def __play_move(self,move, board):
		board[move[0], move[1]] = 1
		dx = [-1,-1,-1,0,1,1,1,0]
		dy = [-1,0,1,1,1,0,-1,-1]
		for i in range(len(dx)):
			if self.__confirm_direction(move,dx[i],dy[i],board):
				self.__change_stones_in_direction(move,dx[i],dy[i],board)

	def PM(self, move, board):
		self.__play_move(move, board)

	def __change_stones_in_direction(self,move,dx,dy, board):
		posx = move[0]+dx
		posy = move[1]+dy
		while (board[posx,posy] != 1):
			board[posx,posy] = 1
			posx += dx
			posy += dy

	def get_all_valid_moves(self, board):
		valid_moves = []
		for x in range(self.board_size):
			for y in range(self.board_size):
				if (board[x,y] == 0) and self.__is_correct_move([x, y], board):
					valid_moves.append( (x, y) )
		
		return valid_moves
	
