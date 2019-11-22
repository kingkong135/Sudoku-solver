from numpy_ringbuffer import RingBuffer
from fuzzywuzzy import process
import numpy as np
import cv2

from tool import Singleton
from neural_network import NeuralNetwork
import sudoku_solving


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

    # For testing
    def write_test(self, sudoku_image):
        for i in range(9):
            for j in range(9):
                block = self.puzzle[i, j]
                if block.number != 0:
                    block.write(sudoku_image, str(block.number))

    def get_existing_numbers(self):
        existing_numbers = []
        for i in range(9):
            for j in range(9):
                block = self.puzzle[i, j]
                if block.number != 0:
                    existing_numbers.append((i, j))

        return existing_numbers

    def as_string(self):
        'Turns the numbers of the sudoku into a string to be read by algorithm'
        string = ''
        array = np.ravel(self.puzzle)
        for guy in array:
            string += str(guy.number)

        return string

    def solve_basic(self):
        'Simply reads the numbers and finds a solution. Printed numbers will be less consistent.'
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
            sudoku_solving.result(string)
            if string in self.already_solved_false:
                solved = False
            else:
                solved = sudoku_solving.solve(string)

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


class Block:
    def __init__(self):
        self.img = None
        self.number = 0

        self.prev_guesses = RingBuffer(capacity=5, dtype=(float, (10)))

        self.fontsize = 0
        self.block_pos = (0, 0)
        self.physical_pos = (0, 0)

        self.n = 0

        # Guesses the number every self.maxtimer frames (10?), to not overuse resources
        self.maxtimer = 10
        self.timer = self.maxtimer - 1

    def update(self, img, block_pos, physical_pos):
        self.img = img
        self.block_pos = block_pos

        top, right, bot, left = physical_pos
        average_dimension = (bot - top + right - left) / 2

        # NOTE edit this for better fontsize, positioning of the number
        self.fontsize = average_dimension / 40
        self.n = average_dimension / 4

        # NOTE edit this for better positioning of the number
        self.physical_pos = (physical_pos[3] + 1 + int(self.fontsize * self.n),
                             physical_pos[2] - int(self.fontsize * self.n))

    def guess_number(self, kind=2, confidence_threshold=0):
        '''
        Uses neural networks to guess the number in the image.
        kind=1 is more primitive, just guesses the image (less reliable)
        kind=2 consumes more memory and CPU but is more reliable (averages out a bunch of guesses)
        '''
        if kind == 1:
            if self.img is None:
                number = 0
            else:
                guy = NeuralNetwork.instance()
                prediction = guy.guess(self.img)
                number = np.argmax(prediction, axis=0)

            self.number = number

        if kind == 2:
            # Guesses every self.maxtimer frames
            self.timer += 1
            if self.timer >= self.maxtimer:
                self.timer = 0

                if self.img is None:
                    self.prev_guesses.appendleft(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
                else:
                    guy = NeuralNetwork.instance()
                    prediction = guy.guess(self.img)
                    self.prev_guesses.appendleft(np.array(prediction))

            m = np.mean(self.prev_guesses, axis=0)
            number = np.argmax(m, axis=0)
            if m[number] > confidence_threshold:
                self.number = number

        return self.number

    def write(self, sudoku_image, text):
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(sudoku_image, text, tuple(self.physical_pos),
                    font, self.fontsize, (0, 0, 255), 1, cv2.LINE_AA)
