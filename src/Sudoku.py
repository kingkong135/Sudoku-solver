import numpy as np
import cv2
from numpy_ringbuffer import RingBuffer

class Sudoku:
    def __init__(self):
        size = (9, 9)
        self.already_solved = {}
        self.already_solved_numbers = {}
        self.already_solved_false = []
        self.puzzle = np.empty(size, dtype=np.obj)
        for i in range(9):
            for j in range(9):
                self.puzzle[i, j ] = Block()











class Block:
    def __init__(self):
        # include position
