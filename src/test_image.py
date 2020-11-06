import cv2
import argparse
from ai import NeuralNetwork
from image_processing import image_processing

# Load model
NeuralNetwork.instance()
# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='an image')
parser.add_argument('output', nargs='?', default='output.png', type=str, help='output file')

args = parser.parse_args()
# read and preprocess one data
filename = args.input
output_file = args.output

img = cv2.imread(filename)
# cv2.imshow('input', img)
output = image_processing(img)
# cv2.imshow("output", output)
cv2.imwrite(output_file, output)

cv2.waitKey(0)
cv2.destroyAllWindows()
