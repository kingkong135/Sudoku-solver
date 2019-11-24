from fuzzywuzzy import process
import numpy as np


from tool import Singleton
import sudoku_solving
from Block import Block

@Singleton
class Sudoku:
    def __init__(self):
        size = (9, 9)
        self.already_solved = {}
        self.already_solved_numbers = {}
        self.already_solved_false = []
        self.puzzle = np.empty(size, dtype=np.object)
        for i in range(size[0]):
            for j in range(size[1]):
                self.puzzle[i, j] = Block()

    def update_block(self, img, block_pos, physical_pos):
        self.puzzle[block_pos].update(img, block_pos, physical_pos)

    def guess_sudoku(self, confidence_threshold=0):
        for i in range(9):
            for j in range(9):
                block = self.puzzle[i, j]
                block.guess_number(confidence_threshold=confidence_threshold)

    def write_solution(self, sudoku_image, solution, ignore=None):
        if solution is not False:
            cols = '123456789'
            rows = 'ABCDEFGHI'
            for i in range(9):
                for j in range(9):
                    number = solution[rows[i] + cols[j]]
                    block = self.puzzle[i, j]
                    if ignore is None:
                        if block.number == 0:
                            block.write(sudoku_image, number)
                    else:
                        if (i, j) not in ignore:
                            block.write(sudoku_image, number)

    def get_existing_numbers(self):
        existing_numbers = []
        for i in range(9):
            for j in range(9):
                block = self.puzzle[i, j]
                if block.number != 0:
                    existing_numbers.append((i, j))

        return existing_numbers

    def as_string(self):
        string = ''
        array = np.ravel(self.puzzle)
        for guy in array:
            string += str(guy.number)

        return string

    def solve_basic(self):
        string = self.as_string()
        if string in self.already_solved.keys():
            return self.already_solved[string]
        else:
            solved = sudoku_solving.solve(string)
            return solved

    def solve_approximate(self, approximate=False):
        'If it finds a sudoku similar to one it has already done, uses its solution'
        string = self.as_string()
        if string in self.already_solved.keys():

            return self.already_solved[string], self.already_solved_numbers[string]
        else:
            # We save the attempts that we already did but were unsuccesful
            if string in self.already_solved_false:
                solved = False
            else:
                solved = sudoku_solving.solve(string)
                # Print answer
                sudoku_solving.result(string)

            # If the sudoku is unsolvable but very similar to one we already did
            # we assume it's the same one but we couldn't quite catch some numbers
            # Approximate is percent-based, 90 = 90%
            if solved is False:
                # Saves this sudoku as false so we don't have to try to solve it every frame
                self.already_solved_false.append(string)

                if self.already_solved.keys():

                    guesses = process.extract(string, self.already_solved.keys())

                    if guesses:

                        # Prioritizes length, then similarity to the guess
                        if approximate is False:
                            best = max(guesses, key=lambda x: (x[1], len(self.already_solved_numbers[x[0]])))[0]
                            return self.already_solved[best], self.already_solved_numbers[best]
                        else:
                            sorty = sorted(guesses, key=lambda x: (len(self.already_solved_numbers[x[0]]), x[1]),
                                           reverse=True)
                            for item in sorty:
                                if item[1] > approximate:
                                    # Sort them by length and then get the one with biggest length that has addecuate ratio?
                                    return self.already_solved[item[0]], self.already_solved_numbers[item[0]]
                            else:
                                best = max(guesses, key=lambda x: (x[1], len(self.already_solved_numbers[x[0]])))[0]
                                return self.already_solved[best], self.already_solved_numbers[best]

            # Only saves correct solutions
            if solved is not False:
                # also save the numbers that already exist in the array
                # (so we don't write over them if we can't see them)
                self.already_solved_numbers[string] = self.get_existing_numbers()
                self.already_solved[string] = solved

                return solved, self.already_solved_numbers[string]

        return False, False

    def solve(self, img_cropped_sudoku, approximate=False):
        solution, existing_numbers = self.solve_approximate(approximate)
        self.write_solution(img_cropped_sudoku, solution, ignore=existing_numbers)
