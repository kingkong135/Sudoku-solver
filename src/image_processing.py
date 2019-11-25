import cv2
import numpy as np
from tool import crop_from_corners, resize_to_square, perspective_transform, blend_non_transparent
from Sudoku import Sudoku


# img = cv2.imread('../images/1.png')
# w, h = img.shape[:2]


def find_sudoku(img, draw_contours=False, test=False):
    '''Finds the biggest object in the image and returns its 4 corners (to crop it)'''
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(gray, (7, 7), 0)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
    # outdir = '../images/Threshold.png'
    # cv2.imshow("1", edges)
    # cv2.imwrite(outdir, edges)

    # Get contours:
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extracting the image of what we think might be a sudoku:
    topbot_edge = (0, img.shape[0] - 1)
    leftright_edge = (0, img.shape[1] - 1)

    if len(contours) > 1:
        conts = sorted(contours, key=cv2.contourArea, reverse=True)

        # Loops through the found objects
        # for something with at least 4 corners and kinda big (>10_000 pixels)
        for cnt in conts:

            epsilon = 0.025 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            if len(cnt) > 3:
                # Gets the 4 corners of the object (assume it's a square)
                top_left     = min(cnt, key=lambda x: x[0, 0] + x[0, 1])
                bot_right = max(cnt, key=lambda x: x[0, 0] + x[0, 1])
                top_right    = max(cnt, key=lambda x: x[0, 0] - x[0, 1])
                bot_left  = min(cnt, key=lambda x: x[0, 0] - x[0, 1])

                corners = (top_left, top_right, bot_left, bot_right)

                # Sometimes it finds 'objects' which are just parts of the screen
                badobj = False
                for corner in corners:
                    if corner[0][0] in leftright_edge or corner[0][1] in topbot_edge:
                        badobj = True

                if badobj is True:
                    continue

                # Test
                if test:
                    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
                    cv2.circle(img, (top_left[0][0], top_left[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                    cv2.circle(img, (top_right[0][0], top_right[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                    cv2.circle(img, (bot_left[0][0], bot_left[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                    cv2.circle(img, (bot_right[0][0], bot_right[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                    outdir = '../images/draws_contours.png'
                    cv2.imwrite(outdir, img)
                    cv2.imshow("draws_contours", img)
            else:

                return edges, None

            # NOTE edit this for different webcams, I found at least size 10k is good
            if cv2.contourArea(cnt) > 10000:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if draw_contours is True:
                    cv2.drawContours(edges, [box], 0, (0, 255, 0), 2)

                # Returns the 4 corners of an object with 4+ corners and area of >10k
                return edges, corners

            else:
                return edges, None
    return edges, None


def build_sudoku(img, test=False):
    # can dilate/open if numbers are small or blur if there's noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.dilate(gray, np.ones((2, 2)))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((1,5),np.uint8))
    # gray = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 7)

    h, w = img.shape[:2]

    # # Sudoku object that will contain all the information
    sudoku = Sudoku.instance()
    k = 0
    border = 4
    x = w/9
    y = h/9
    ans = []
    for i in range(9):
        for j in range(9):
            top = int(round(y * i + border))
            left = int(round(x * j + border))
            right = int(round(x * (j + 1) - border))
            bot = int(round(y * (i + 1) - border))
            if i == 0:
                top += border
            if i == 8:
                bot -= border
            if j == 0:
                left += border
            if j == 8:
                right -= border

            point = [
                [[left, top]],
                [[right, top]],
                [[left, bot]],
                [[right, bot]]
            ]

            square, _ = crop_from_corners(edges, point)
            # cv2.imshow('test ' + str((i+1)*(j+1)), square)
            if test is True:
                if i == 0 and j == 3:
                    cv2.imshow('square', square)
                if i == 1 and j == 0:
                    cv2.imshow('ss', square)

            fat_square = square.copy()
            contours, _ = cv2.findContours(fat_square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(fat_square, contours, -1, (255, 255, 255), 2)
            # Get the contour of the number (biggest object in a case)
            contours, _ = cv2.findContours(fat_square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            physical_position = [top, right, bot, left]



            if contours:
                conts = sorted(contours, key=cv2.contourArea, reverse=True)
                # Get the biggest object in the case (assume it's a number)
                cnt = conts[0]

                # minarea is an arbitrary size that the number must be to be considered valid
                # NOTE change it if it detects noise/doesn't detect numbers (0.04)
                minarea = x * y * 0.04
                if cv2.contourArea(cnt) > minarea:
                    # Crop out the number

                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    minx = max(min(box, key=lambda g: g[0])[0], 0)
                    miny = max(min(box, key=lambda g: g[1])[1], 0)
                    maxx = min(max(box, key=lambda g: g[0])[0], int(x))
                    maxy = min(max(box, key=lambda g: g[1])[1], int(y))

                    number_image = square[miny:maxy, minx:maxx]

                    if number_image is None or number_image.shape[0] < 4 or number_image.shape[1] < 4:
                        sudoku.update_block(None, (i, j), physical_position)
                    else:
                        # If we get a valid number image:
                        # Resize it to 28x28 for neural network purposes
                        final = resize_to_square(number_image)
                        sudoku.update_block(final, (i, j), physical_position)
                        k = k + 1
                else:
                    sudoku.update_block(None, (i, j), physical_position)
            else:
                sudoku.update_block(None, (i, j), physical_position)
        # print(k) ## print number in the sudoku
    return sudoku

def image_processing(img):
    img_ans = img
    w, h = img.shape[:2]
    edges, corners = find_sudoku(img, False, False)
    if corners is not None:
        # We crop out the sudoku and get the info needed to paste it back (matrix)
        img_crop, transformation = crop_from_corners(img, corners)
        cv2.imshow('crop', img_crop)
        # cv2.imwrite('../images/crop.png', img_crop)

        transfor_matrix = transformation['matrix']
        original_shape = transformation['original_shape']

        # inverse the matrix for we can thuc hien chuyen doi sau
        transfor_matrix = np.linalg.pinv(transfor_matrix)
        sudoku = build_sudoku(img_crop, test=False)
        sudoku.guess_sudoku(confidence_threshold=0)
        sudoku.solve(img_crop, approximate=0.90)

        img_sudoku_final = perspective_transform(h, w, img_crop, transfor_matrix, original_shape)
        # cv2.imshow("img_sudoku_final", img_sudoku_final)
        # cv2.imwrite('../images/perspective_transform.png', img_sudoku_final)

        img_ans = blend_non_transparent(img, img_sudoku_final)
        # cv2.imshow('crop2', img_ans)
        # cv2.imwrite('../images/blend_non_transparent.png', img_final)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_ans