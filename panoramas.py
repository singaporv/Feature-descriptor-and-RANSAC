import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    # M = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    warp_im = cv2.warpPerspective(im2,H2to1,(1500,600))
    im1 = cv2.warpPerspective(im1,np.eye(3),(1500,600))

    # Different types of blending 
    # pano_im = cv2.addWeighted( im1, 0.5, warp_im, 0.5, 0.0)
    pano_im = np.maximum(im1, warp_im)

    # pano_im = np.mean([im1, warp_im])


    np.save("../results/q6_1",[H2to1])

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...

    first_1 = np.array([[576],[1064],[1]])
    first_2 = np.array([[0],[0],[1]])
    first_3 = np.array([[0],[1064],[1]])
    first_4 = np.array([[576],[0],[1]])

    second_1 = np.matmul(H2to1,first_1)
    # print (second)
    n4 = second_1[2,0]
    second_1 = second_1/n4
    print (second_1)

    second_2 = np.matmul(H2to1,first_2)
    # print (second)
    n4 = second_2[2,0]
    second_2 = second_2/n4
    print (second_2)

    second_3 = np.matmul(H2to1,first_3)
    # prin_t (second)
    n4 = second_3[2,0]
    second_3 = second_3/n4
    print (second_3)

    second_4 = np.matmul(H2to1,first_4)
    # print (second)
    n4 = second_4[2,0]
    second_4 = second_4/n4
    print (second_4)

    # Through geometry
    image_width = second_1[1] - second_4[1]
    image_height = second_4[0] - second_3[0]

    # 600 and 1500 are previously defined distances in the function (Random values taken)
    translation_height = image_height - 600
    translation_width = 1500 - image_width 


    # im1 = cv2.warpPerspective(im1,np.eye(3),(3000,600))

    M = np.eye(3) + np.array([[.85,0,translation_height],[0,.85,400],[0,0,1]])
    warp_im_1 = cv2.warpPerspective(im1,M,(1714,815))
    warp_im_2 = cv2.warpPerspective(im2,np.matmul(M,H2to1),(1714,815))
    pano_im = np.maximum(warp_im_1, warp_im_2)


    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pano_im

def generatePanorama(im1,im2):

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches, m = briefMatch(desc1, desc2)

    m,n = matches.shape
    H2to1 = ransacH(matches, m, locs1, locs2, num_iter=5000, tol=2)


    # To get the M - Translation and scalling matrix
    first_1 = np.array([[1064],[576],[1]])
    first_2 = np.array([[0],[0],[1]])
    first_3 = np.array([[0],[576],[1]])
    first_4 = np.array([[1064],[0],[1]])

    second_1 = np.matmul(H2to1,first_1)
    n4 = second_1[2,0]
    second_1 = second_1/n4
    print (second_1)

    second_2 = np.matmul(H2to1,first_2)
    n4 = second_2[2,0]
    second_2 = second_2/n4
    print (second_2)

    second_3 = np.matmul(H2to1,first_3)
    n4 = second_3[2,0]
    second_3 = second_3/n4
    print (second_3)

    second_4 = np.matmul(H2to1,first_4)
    n4 = second_4[2,0]
    second_4 = second_4/n4
    print (second_4)

    # Through geometry
    image_width = second_1[1] - second_4[1]
    image_height = second_4[0] - second_3[0]

    # 600 and 1500 are previously defined distances in the function (Random values taken)
    translation_height = image_height - 600
    translation_width = 1500 - image_width 


    # im1 = cv2.warpPerspective(im1,np.eye(3),(3000,600))

    M = np.eye(3) + np.array([[.85,0,translation_height],[0,.85,400],[0,0,1]])

    warp_im_1 = cv2.warpPerspective(im1,M,(1714,815))
    warp_im_2 = cv2.warpPerspective(im2,np.matmul(M,H2to1),(1714,815))
    im3 = np.maximum(warp_im_1, warp_im_2)


    return im3




if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    print(im2.shape)
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches, m = briefMatch(desc1, desc2)

    # np.save("temp_pana",[locs2, locs1, desc2, desc1, matches])
    locs2, locs1, desc2, desc1, matches = np.load("temp_pana.npy")
    # print (locs1.shape)
    # print (locs2.shape)
    # print (matches.shape)

    m,n = matches.shape

    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, m, locs1, locs2, num_iter=5000, tol=2)
    np.save('../results/H2to1.npy',H2to1)
    
    # pano_im = imageStitching(im1, im2, H2to1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    # cv2.imwrite('../results/panoImg.png', pano_im)

    # im3 = generatePanorama(im1,im2)

    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()