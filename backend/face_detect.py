import os 
import sys
import cv2
import copy
import subprocess
import csv
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import io
import matplotlib.pyplot as plt
from scipy import misc
import time

def realtime():
    ESC_KEY = 27     
    INTERVAL= 33     
    FRAME_RATE = 30
    i = 0
    j = 0
    b_num = 0
    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    cascade_file = '/usr/local/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml'
    cascade = cv2.CascadeClassifier(cascade_file)

    cap = cv2.VideoCapture(DEVICE_ID)

    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    while end_flag == True:

        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))
        if (b_num != len(face_list)):
            b_num = len(face_list)
            take_pic = 1
        
        for (x, y, w, h) in face_list:
            
            color = (0, 0, 225)
            pen_w = 3
            cv2.rectangle(img_gray, (x, y), (x+w, y+h), color, thickness = pen_w)
            start = time.time()
            
            if (take_pic):
                cv2.imwrite('templates/images/image.jpg', img)
                take_pic = 0

        cv2.imshow(ORG_WINDOW_NAME, c_frame)
        cv2.imshow(GAUSSIAN_WINDOW_NAME, img_gray)

        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        end_flag, c_frame = cap.read()

    cv2.destroyAllWindows()
    cap.release()

def take():
    in_path = 'templates/images/image.jpg'
    out_path = 'templates/images/face.jpg'

    image = cv2.imread(in_path)
    cv2.imwrite(out_path, image)

    print ('[image_take.py] is Sucess')

def detect():
    
    in_path = 'templates/images/face.jpg'
    out_path = 'templates/images/face_detect.jpg'

    HAAR_FILE = '/usr/local/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml'
    cascade = cv2.CascadeClassifier(HAAR_FILE)
    
    img_color = cv2.imread(in_path)
    img = cv2.imread(in_path, 0)
    face = cascade.detectMultiScale(img)
    
    for x,y,w,h in face:
        face_cut = img_color[y:y+h, x:x+w]
        cv2.imwrite(out_path, face_cut)
    print ('[face_detect.py] is Sucess')

def prepro():

    in_path = 'templates/images/face_detect.jpg'
    out_path = 'templates/images/prepro.jpg'

    img = cv2.imread(in_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)
    cv2.imwrite(out_path, equ)

    print ("[prepro.py] is Sucess")

def feature():

    in_path = 'templates/images/prepro.jpg'
    image_list = []
    feature = []

    desc = LocalBinaryPatterns(8, 1)
    image = cv2.imread(in_path, 0)
    div(image)
    file_names = sorted(os.listdir('templates/images/div/'))

    for file in file_names:
        image_list.append(file)

    for i in range(len(image_list)):
        img = cv2.imread('templates/images/div/' + image_list[i], 0)
        hist = desc.describe(img)
        string = ''
        for i in hist:
            hstring = str(i)
            hstring = round((float(hstring)),8)
            string = str(hstring)
            feature.append(string)
    print ("[feature.py] is Sucess")
    return feature

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius
        
    def describe(self, image, eps=1e-7):
        lbp_list = []
        lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        lbp_list.append(lbp)
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2), density=True)
        hist = hist.astype("float")
        return hist

def div(img):
    height, width = img.shape
    height_split = 8
    width_split = 8
    new_img_height = int(height / height_split)
    new_img_width = int(width / width_split)
    
    for h in range(height_split):
        height_start = h * new_img_height
        height_end = height_start + new_img_height
        for w in range(width_split):
            width_start = w * new_img_width
            width_end = width_start + new_img_width
            file_name = 'test_' + str(h) + '_' + str(w) + '.jpg'
            clp = img[height_start:height_end, width_start:width_end]
            cv2.imwrite('templates/images/div/' + file_name, clp)

if __name__ == '__main__':
    image_take()
    face_detect()
    prepro()
    feature()