from numpy_ringbuffer import RingBuffer
import numpy as np
import cv2
from ai import NeuralNetwork

class Block:
    def __init__(self):
        self.img = None
        self.number = 0

        self.prev_guesses = RingBuffer(capacity=5, dtype=(float, (10)))

        self.fontsize = 0
        self.block_pos = (0, 0)
        self.physical_pos = (0, 0)

        self.n = 0

        # Guesses the number every self.maxtimer frames (10), to not overuse resources
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
