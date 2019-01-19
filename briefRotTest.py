import BRIEF
import numpy as np
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt 



def rot10(im1,rows,cols, angle):
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	im_r = cv2.warpAffine(im2,M,(cols, rows))
	return im_r

if __name__ == '__main__':
	im1 = cv2.imread('../data/model_chickenbroth.jpg')
	im2 = cv2.imread('../data/chickenbroth_01.jpg')
	rows,cols,l = im1.shape 
	angles = []
	match_num = []
	for it in range(0,361,10):
		angles.append(it)
		im_r = rot10(im1, rows, cols, it)
		it +=10

	# print (angles)

		# cv2.imshow('im_r',im_r)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# compareX, compareY = BRIEF.makeTestPattern()
		# locs1, desc1 = BRIEF.briefLite(im1)
		# locs2, desc2 = BRIEF.briefLite(im_r)
		# matches,m = BRIEF.briefMatch(desc1, desc2)
		# # ##BRIEF.plotMatches(im1,im_r,matches,locs1,locs2)
		# match_num.append(m)


	# print (match_num)
	# result
	match_num = [149, 75, 50, 77, 37, 48, 37, 43, 41, 56, 57, 64, 37, 39, 38, 43, 51, 47, 46, 43, 53, 44, 51, 51, 51, 43, 41, 38, 58, 48, 42, 46, 35, 42, 50, 82, 148]


	# Histogram plot
	plt.title("Angle vs matches")
	plt.bar(angles, match_num, width = 5, align='center')
	plt.xticks(angles) #Replace default x-ticks with xs, then replace xs with labels
	plt.yticks(match_num)
	plt.show()