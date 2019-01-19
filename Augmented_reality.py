import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import computeH
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt 

def compute_extrinsics(K,H):

	H_dash = np.matmul(np.linalg.inv(K), H)
	u,s,v = np.linalg.svd(H_dash[:, :2])

	temp = np.matrix([[1,0],[0,1],[0,0]])

	R = np.zeros((3,3))
	R[:, :2] = np.matmul(u, np.matmul(temp,v))
	b = np.cross(R[:,0],R[:,1])
	R[:, 2] = b

	if (np.linalg.det(R) == -1): 
		R[:, 2] = R[:, 2]*-1

	D = np.divide(H_dash[:,:2],R[:,:2])

	lam = np.sum(D)/6

	t = H_dash[:,2]/lam

	return R,t


def project_extrinsics(K,W,R,t):

	W[0] = W[0] + 6.3
	W[1] = W[1] + 18.4
	W[2] = W[2] - 6.8581/2
	l = np.matmul(K,np.matmul(R, W)+t)
	# Normalize
	l = l/l[2]

	return l


if __name__ == '__main__':

	im = cv2.imread("../data/prince_book.jpeg")
	Wdash = np.loadtxt("../data/sphere.txt")
	W = np.matrix([[0.0, 18.2, 18.2, 0.0],[0.0, 0.0, 26.0, 26.0],[0.0,0.0,0.0,0.0]])
	X = np.matrix([[483,1704,2175,67],[810,781,2217,2286]])
	K = np.matrix([[3043.72,0.0,1196.00],[0.0,3043.72,1604.0],[0.0,0.0,1.0]])


	H = computeH(X,W)

	R,t = compute_extrinsics(K,H)
	P = project_extrinsics(K,Wdash,R,t)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	plt.imshow(im)
	plt.plot(P[0],P[1], 'y.')
	plt.show()

