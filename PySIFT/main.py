from skimage.io import imread
from sift import SIFT
import cv2
import pickle
from os.path import isfile

if __name__ == '__main__':
	num_img = 3

	kp_pyrs = []
	ims = []

	im = imread('C:/Users/Shantanu/img2.jpg')
	im = cv2.resize(im, (1268,720))
	ims.append(im)

	if isfile('C:/Users/Shantanu/kp_pyr.pkl'):
		kp_pyrs.append(pickle.load(open('C:/Users/Shantanu/kp_pyr.pkl', 'rb')))
	

	print('Performing SIFT on image')

	sift_detector = SIFT(im)
	_ = sift_detector.get_features()
	kp_pyrs.append(sift_detector.kp_pyr)

	pickle.dump(sift_detector.kp_pyr, open('C:/Users/Shantanu/kp_pyr.pkl', 'wb'))
	pickle.dump(sift_detector.feats, open('C:/Users/Shantanu/feat_pyr.pkl', 'wb'))

	import matplotlib.pyplot as plt
	

	_, ax = plt.subplots(1, 1)
	ax.imshow(ims[0])

	kps = kp_pyrs[0][0]*(2**0)
	ax.scatter(kps[:,0], kps[:,1], c='b', s=2)
	#plt.show()
	plt.savefig('C:/Users/Shantanu/plot.png')


