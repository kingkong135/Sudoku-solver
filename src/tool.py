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

    # truc tiep xoay cac hinh chu nhat cheo ve hcn thang
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

# Bien doi quan diem
def perspective_transform(h, w, img, transformation_matrix, original_shape=None):
    warped = img
    if original_shape is not None:
        if original_shape[0] > 0 and original_shape[1] > 0:
            warped = cv2.resize(warped, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)

    white_image = np.zeros((h, w, 3), np.uint8)

    white_image[:, :, :] = 255

    # warped = cv2.warpPerspective(warped, transformation_matrix, (640, 480), borderMode=cv2.BORDER_TRANSPARENT)
    warped = cv2.warpPerspective(warped, transformation_matrix, (h, w))

    return warped

# pha tron khong trong suot
def blend_non_transparent(face_img, overlay_img):
    # Let's find a mask covering all the non-black (foreground) pixels
    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)[1]

    # Let's shrink and blur it a little to make the transitions smoother...
    overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))

    # And the inverse mask, that covers all the black (background) pixels
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def crop_minAreaRect(src, rect):
    # Get center, size, and angle from rect
    center, size, theta = rect

    # Angle correction
    if theta < -45:
        theta += 90

    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D(center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))
    out = cv2.getRectSubPix(dst, size, center)
    return out


def resize_to_square(img, goal_dimension=28, border=2):
    h, w = img.shape[0], img.shape[1]

    ratio = goal_dimension / max(h, w)

    color = [0, 0, 0]
    constant = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=color)
    background = np.zeros((goal_dimension, goal_dimension), dtype=np.int)
    resized = cv2.resize(constant, (int(round(w * ratio)), int(round(h * ratio))),
                         interpolation=cv2.INTER_AREA)

    x_offset = (goal_dimension - resized.shape[1]) // 2
    y_offset = (goal_dimension - resized.shape[0]) // 2

    background[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

    final = background
    return np.uint8(final)


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)
