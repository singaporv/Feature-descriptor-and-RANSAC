import numpy as np
import cv2
import matplotlib.pyplot as plt 

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    # #checking the shape of the matrix
    # print (im_pyramid.shape)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    for i in range(len(levels)-1):
        single_l = gaussian_pyramid[:,:,i]-gaussian_pyramid[:,:,i+1]
        DoG_pyramid.append(single_l)

    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    # print (DoG_pyramid.shape)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''

    H,W, gauss_levels = DoG_pyramid.shape
    R = np.zeros([H,W,gauss_levels])
    levels = levels=[-1,0,1,2,3,4]
    for i in range(len(levels)-1): 
        H_x = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F, 1,0) #check between 64 and 32 bits
        H_xx = cv2.Sobel(H_x,cv2.CV_64F, 1,0)

        H_y = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F, 0,1)
        H_yy = cv2.Sobel(H_y,cv2.CV_64F, 0,1)

        H_xy = cv2.Sobel(H_y,cv2.CV_64F, 1,0)

        for h in range(H):
            for w in range(W):
                trace = H_xx[h,w]+H_yy[h,w]
                det = (H_xx[h,w]*H_yy[h,w])-(H_xy[h,w]*H_xy[h,w])

                if(det == 0):
                    R[h,w,i] = 0
                else:
                    R[h,w,i] = (trace**2)/det

    principal_curvature = R
    # print(principal_curvature.shape)

    #principal_curvature = np.stack(principal_curvature, axis=-1)

    # print (principal_curvature.shape)
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    # locsDoG = None
    th_contrast = 0.03
    th_r = 12

    neighbors = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
                               for y2 in range(y-1, y+2)
                               if (-1 < x <= X and
                                   -1 < y <= Y and
                                   (x != x2 or y != y2) and
                                   (0 <= x2 <= X) and
                                   (0 <= y2 <= Y))]
    # Parameters of pyramid
    X, Y, Z = DoG_pyramid.shape[0], DoG_pyramid.shape[1], DoG_pyramid.shape[2]
    locsDoG = []
    for e in range(Z):
        for f in range(X-1):
            for g in range(Y-1):
                temp = neighbors(f,g)
                # print (temp)
                # print (len(temp))
                t=[]
                for i in range(len(temp)):
                    t.append(DoG_pyramid[temp[i][0], temp[i][1], e])
                if (e==0):
                    t.append(DoG_pyramid[f,g,e+1])
                elif (e==Z-1):
                    t.append(DoG_pyramid[f,g,e-1])
                else:
                    t.append(DoG_pyramid[f,g,e-1])
                    t.append(DoG_pyramid[f,g,e+1])
                t=np.array(t)
                t_max = np.amax(t)
                t_min = np.amin(t)
                if (DoG_pyramid[f,g,e]>=t_max or DoG_pyramid[f,g,e]<=t_min):
                    if(np.abs(DoG_pyramid[f,g,e]) > th_contrast and np.abs(principal_curvature[f,g,e]) < th_r):
                        locsDoG.append([g,f,e])
    # print (len(locsDoG))

    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################

    gauss_pyramid = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    im = plt.imread('../data/model_chickenbroth.jpg')
    implot = plt.imshow(im)
    color = (0,0,100)
    area = np.pi*5
    
    for i in range(len(locsDoG)):
        x = locsDoG[i][0]
        y = locsDoG[i][1]
        plt.scatter(x, y, s=area, c="r", alpha=0.5)
    

    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    locsDoG = np.asarray(locsDoG)

    return locsDoG, gauss_pyramid




if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    # locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    print (locsDoG.shape)


