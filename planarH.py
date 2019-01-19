import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...

    n = p1.shape[1]
    A = np.empty([2*n,9])
    for i in  range(0,2*n,2):
        A[i,0] = 0
        A[i,1] = 0
        A[i,2] = 0
        A[i,3] = - p2[0,i//2]
        A[i,4] = - p2[1,i//2]
        A[i,5] = - 1
        A[i,6] = p1[1,i//2]*p2[0,i//2]
        A[i,7] = p1[1,i//2]*p2[1,i//2]
        A[i,8] = p1[1,i//2]
        A[i+1,0] = p2[0,i//2]
        A[i+1,1] = p2[1,i//2]
        A[i+1,2] = 1
        A[i+1,3] = 0
        A[i+1,4] = 0
        A[i+1,5] = 0
        A[i+1,6] = - p1[0,i//2]*p2[0,i//2]
        A[i+1,7] = - p1[0,i//2]*p2[1,i//2]
        A[i+1,8] = - p1[0,i//2]


    # print (A.astype("int"))
    u,s,vT = np.linalg.svd(A)
    v = vT.T

    H2to1 = v[:,-1]
    
    # normalising array
    H2to1 = H2to1.reshape(3,3)
    n1 = H2to1[2][2]
    H2to1 = H2to1/n1
    # print (H2to1)

    return H2to1

def ransacH(matches,m, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...

    # m is the number of matches
    # computing p1 and p2
    p1 = []
    p2 = []
    for i in range(m):
        p1.append(locs1[matches[i,0],:2])
        p2.append(locs2[matches[i,1],:2])

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    # taking transpose of it
    p1 = p1.T
    p2 = p2.T


    # ### Question 4

    # p1 = p1[:,:4]
    # p2 = p2[:,:4]


    # p1 = np.asarray(p1)
    # p2 = np.asarray(p2)



    # H2to1 = computeH(p1,p2)


    # p1_H_test = np.asarray([p1[0,0], p1[1,0],1])

    # # print(p1_H_test)

    # p2_H_test = [p2[0,0], p2[1,0],1]
    # p2_H_test = np.reshape(p2_H_test,(3,1))
    # # print (p2_H_test)



    # H_inv= np.linalg.inv(H2to1)
    # n2 = H_inv[2,2]
    # H_inv = H_inv/n2


    # ep2 = np.matmul(H_inv, p1_H_test)


    # n3 = ep2[2]
    # ep2 = ep2/n3
    # print (ep2)
    # print (p2_H_test)






    # Ransac - Ques 5

    Max_inliers = 0

    for i in range(num_iter):

        rand = np.random.randint(m, size = 4)
        # print (rand)
        p1_R = p1[:,rand]
        p2_R = p2[:,rand]


        # H2to1 = computeH(p1,p2)
        H2to1 = computeH(p1_R,p2_R)

        # print (H2to1.shape)
        # print (H2to1)

        # p1_H_test = [p1[0,0], p1[1,0],1]
        # p1_H_test = np.reshape(p1_H_test,(3,1))

        p1_H_test = np.vstack([p1,[1]*m])


        # print ("For testing p2_H is: ")
        p2_H_test = np.vstack([p2,[1]*m])
        # print (p2_H_test)



        # # To test

        H2to1_inv = np.linalg.inv(H2to1)
        n2 = H2to1_inv[2,2]
        H2to1_inv = H2to1_inv/n2
        ep2 = np.matmul(H2to1_inv, p1_H_test)


        # normalising array
        n3 = ep2[2, :]
        ep2 = ep2/n3
        # print (ep2)
        # print (p2_H_test)
        error = ep2.T - p2_H_test.T

        d = np.linalg.norm(error, axis=1)
        # print (d)

        inliers = 0
        for z in d:
            if (z<tol):
                inliers +=1
        # print (inliers)
                    
        if (inliers > Max_inliers):
            Max_inliers = inliers
            bestH = H2to1


# print (Max_inliers)
    return bestH



        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches, m = briefMatch(desc1, desc2)
    # m is the number of matches

    ransacH(matches,m, locs1, locs2, num_iter=5000, tol=2)

