import cv2

from ai import NeuralNetwork
from image_processing import image_processing

# Load model
NeuralNetwork.instance()

img = cv2.imread('../images/1.png')
output = image_processing(img)
cv2.imshow("output", output)

cv2.waitKey(0)
cv2.destroyAllWindows()