import cv2 
import os
import shutil
import numpy as np 
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import os
import sys
from skimage.io import imread
from sift import SIFT
import cv2
import pickle
from os.path import isfile


# input comes from STDIN
for line in sys.stdin:
    line = line.strip()
    kp_pyrs = []
    ims = []

    im = imread(line)
    im = cv2.resize(im, (1268,720))
    ims.append(im)

    list_string = line.split('/')
    list_string_name = list_string[-1].split('.')
    y = '/'.join(list_string[:-1])
    y = y+'/'
    
    kp_pyr_file = 'C:/Users/Shantanu/new_video/'+list_string_name[0]+'_kp_pyr.pkl'

    if isfile(kp_pyr_file):
            kp_pyrs.append(pickle.load(open(kp_pyr_file, 'rb')))
    

    #print('Performing SIFT on image')

    fear_pyr_file = 'C:/Users/Shantanu/new_video/'+list_string_name[0]+'_feat_pyr.pkl'

    sift_detector = SIFT(im)
    _ = sift_detector.get_features()
    kp_pyrs.append(sift_detector.kp_pyr)

    pickle.dump(sift_detector.kp_pyr, open(kp_pyr_file, 'wb'))
    pickle.dump(sift_detector.feats, open(fear_pyr_file, 'wb'))

    import matplotlib.pyplot as plt
    

    _, ax = plt.subplots(1, 1)
    ax.imshow(ims[0])

    kps = kp_pyrs[0][0]*(2**0)
    ax.scatter(kps[:,0], kps[:,1], c='b', s=2)
    #plt.show()

    scatter_plot = y+list_string_name[0]+'_plot.png'
    plt.savefig(scatter_plot)
    print(scatter_plot+"\n")
    
