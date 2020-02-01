import cv2 
import os
import shutil
import numpy as np 
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import os
import sys


for line in sys.stdin:
    line = line.strip()

    cam = cv2.VideoCapture(line) 

    list_string = line.split('/')
    list_string_name = list_string[-1].split('.')
    y = '/'.join(list_string[:-1])
    y = y+'/'

    if os.path.exists(y+list_string_name[0]):
        shutil.rmtree(y+list_string_name[0])

    os.makedirs(y+list_string_name[0]) 

    currentframe = -1
  
    while(1):
        ret,frame = cam.read()
        if not ret:
            break
        else:
            currentframe += 1
            if currentframe%30 == 0:
                name = y+list_string_name[0]+'/frame' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame)
                print(name)
            

    # Release all space and windows once done 
    
    cam.release() 
    cv2.destroyAllWindows()
