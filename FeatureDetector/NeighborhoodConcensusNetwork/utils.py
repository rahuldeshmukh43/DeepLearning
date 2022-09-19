import numpy as np
import cv2
import matplotlib.cm as cm
import logging


radius = 1
# color=(0,255,0) #green BGR
thickness=-1
line_size=2

def draw_matches(imA, imB, kps_A, predicted_matches_B):
    """
    draws keypoints in image A and B
    :param imA image A np array HWC  RGB
    :param imB image B np array HWC RGB
    :param kps_A keypoints in imA np array [num_kps,2] #(x,y) 
    :param predicted_matches in imB. numpy array [num_kps,2] #row, col (y,x)
    :return: img np array CHW RGB
    """    
    H1,W1 = imA.shape[:-1]
    Npts = kps_A.shape[0]
    img = np.concatenate((imA, imB), 1)
    img = np.ascontiguousarray(img)#[:,:,::-1] #cv2 needs BGR
    color = cm.rainbow(np.linspace(0, 1, Npts))
    for ic in range(Npts):
        x0, y0= kps_A[ic,:]
        x1, y1 = predicted_matches_B[ic][:] + np.array([W1, 0]) #offset added
        x0, y0, x1, y1 = np.array([x0, y0, x1, y1]).astype(np.int32)
        img = cv2.circle( img, (x0,y0), radius, color[ic], thickness, cv2.LINE_AA)
        img = cv2.circle( img,(x1,y1), radius, color[ic], thickness, cv2.LINE_AA)
        img = cv2.line(img, (x0,y0), (x1,y1), color[ic], line_size,cv2.LINE_AA)
    #HWC to CHW
    # img = img[:,:,::-1] #BGR to RGB
    img = img.transpose(2,0,1)
    return img

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def Setup_Logger(name:str, log_file:str, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

