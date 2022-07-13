import cv2
import numpy as np
import time

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

capCamera = cv2.VideoCapture(2)
capVideo = cv2.VideoCapture(0)
capCamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capCamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
capVideo.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capVideo.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
time.sleep(3)

while True:
    isNextFrameAvail1, frame1 = capCamera.read()
    isNextFrameAvail2, frame2 = capVideo.read()
    if not isNextFrameAvail1 or not isNextFrameAvail2:
        continue
    frame2Resized = cv2.resize(frame2,(frame1.shape[0],frame1.shape[1]))

    # ---- Option 1 ----
    numpy_vertical = np.vstack((frame1, frame2))
    #numpy_horizontal = np.hstack((frame1, frame2))

    # ---- Option 2 ----
    #numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
    #numpy_horizontal_concat = np.concatenate((frame1, frame2), axis=1)

    frame_resized = image_resize(numpy_vertical, 300, 300)
    frame_resized = cv2.resize(frame_resized, (300, 300))

    cv2.imwrite("dubelcam.jpg", frame_resized)
    time.sleep(1)
