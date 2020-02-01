#!/usr/bin/env python
"""reducer.py"""

import cv2 
import os
import shutil
import numpy as np 
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import os
import sys



# input comes from STDIN
for line in sys.stdin:
    line = line.strip()
    #print(line)
    #new_loc_hdfs = 'C:/Users/Shantanu/'+list_string_name[0]+'/frame60.jpg'
    img = cv2.imread(line, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1268,720))
    ret,thresh1 = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(thresh1,-1,kernel)
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
    img1 = Image.fromarray(dst)
    text = pytesseract.image_to_string(img1)
    if(text != ""):
        text = text.strip()
        text = text.split("\n")
        for t in text:
            if(t != "" and len(t)>5):
                print(t.encode("utf-8"))
        print('\n')
    
    

