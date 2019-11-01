import cv2
import numpy as np

def crop_from_corners(img, corners, make_square=True):
    cnt = np.array([corners[0], corners[1], corners[2], corners[3]])
    rect = cv2.minAreaRect(cnt)
    center, size, theta = rect
    # dieu chinh goc
    if theta < -45:
        theta += 90

    rect = (center, size, theta)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (0,0,255), 2)

    # get width and height of the detected rectangle
    w = int(rect[1][0])
    h = int(rect[1][1])

    src_pts = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    #  the perspective transformation matrix (ma tran bien doi phoi canh)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (w, h))

    # Making it square for the numbers are more readable
    if make_square:
        try:
            warped = cv2.resize(warped, (max(w, h), max(w, h)), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(e)

    transfor_data = {
        'matrix': M,
        'original_shape': (h, w)
    }
    return warped, transfor_data


def resize_to_square(image, goal_dimension=28, border=2):
    height, width = image.shape[0], image.shape[1]
    smol = max(height, width)

    proportion = goal_dimension / smol

    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=BLACK)
    background = np.zeros((goal_dimension, goal_dimension), dtype=np.int)
    resized = cv2.resize(constant, (int(round(width * proportion)), int(round(height * proportion))),
                         interpolation=cv2.INTER_AREA)

    x_offset = (goal_dimension - resized.shape[1]) // 2
    y_offset = (goal_dimension - resized.shape[0]) // 2

    background[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

    final = background
    return np.uint8(final)

