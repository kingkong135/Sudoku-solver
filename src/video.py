import cv2

from ai import NeuralNetwork
from image_processing import image_processing

# Capture from camera
cap = cv2.VideoCapture(0)
cv2.startWindowThread()

# Load model
NeuralNetwork.instance()

while True:
    _, img = cap.read()
    output = image_processing(img)
    cv2.imshow("output", output)

    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.VideoCapture(0).release()
cv2.waitKey(1)