import cv2, imutils
import numpy as np
import random

def transformations(src, choice):
    if choice == 0:
        # clockwise rotate 90
        dst = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    if choice == 1:
        # counterclockwise rotate 90
        dst = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == 2:
        # rotate 180
        dst = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
    return dst

def flip(src, flipCode):
    return cv2.flip(src, flipCode=flipCode)

def random_crop(img, dstSize, center=False):
    srcH, srcW = img.shape[:2]
    dstH, dstW = dstSize
    if srcH < dstH or srcW < dstW:
        return cv2.resize(img, (dstW, dstH))
    if center:
        y0 = int((srcH - dstH)/2)
        x0 = int((srcW - dstW)/2)
    else:
        y0 = random.randrange(0, srcH - dstH)
        x0 = random.randrange(0, srcW - dstW)
    return img[y0:y0+dstH, x0:x0+dstW,...]

def expand_chan(src):
    channels = cv2.split(src)
    channels.append(sum(channels)/len(channels))
    dst = cv2.merge(channels)
    return dst


def randomShiftScaleRotate(image,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))


    return image