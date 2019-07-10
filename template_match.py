#! /usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt

import os

def find_match(filename):
    print filename
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('images/markers.png',0) # queryImage
    img2 = cv2.imread(filename,0) # trainImage
    

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()# xfeatures2d_SIFT()# .SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        inv_M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
        transformed = cv2.warpPerspective(img2, inv_M, tuple(reversed(img1.shape)))
        plt.imshow(transformed, 'gray')
        basename = os.path.basename(filename)
        basename = basename.replace('jpg', 'png')
        ofile = 'images/inverse-{}'.format(basename)
        plt.savefig(ofile)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray')
    basename = os.path.basename(filename)
    basename = basename.replace('jpg', 'png')
    ofile = 'images/features-{}'.format(basename)
    plt.savefig(ofile)

def main():
    #find_match('images/scene1.jpg')
    #find_match('images/scene2.jpg')
    #find_match('images/scene3.jpg')
    #find_match('images/scene4.jpg')
    find_match('images/scene5.jpg')

if __name__ == '__main__':
    main()
