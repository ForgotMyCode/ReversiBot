import numpy as np

class TranspositionTable:
	FLAG_UNSET = 0
	FLAG_EXACT = 1
	FLAG_LOWERBOUND = 2
	FLAG_UPPERBOUND = 3

	class TranspositionTableEntry:
		def __init__(self, value, depth, flag: int):
			self.value = value
			self.flag = flag
			self.depth = depth

	def __init__(self):
		self.__table = {}

	def get_key(raw_key: np.ndarray):
		return raw_key.tobytes()

	def lookup(self, key) -> TranspositionTableEntry:
		return self.__table.get(key)

	def store(self, key, value: TranspositionTableEntry) -> None:
		self.__table[key] = value

	