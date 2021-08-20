
# Basic Program Of Advance Bar Code


import cv2  #python library used for computer vision
import sys  #module which provides various function and variable
import cv2.aruco as aruco #Module used for detection of object
import numpy as np  #Module Used to perform mathemetical operations on arrays(=array is collection of item stored)


cap = cv2.VideoCapture(2)

im_src = cv2.imread('opencv_logo_with_text.png')

while cap.isOpened():

    
    ret, frame = cap.read()
    scale_percent = 60 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    org_frame = frame
    
    if not ret:
        continue

    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if np.all(ids != None):
        for c in corners :
            x1 = (c[0][0][0], c[0][0][1]) 
            x2 = (c[0][1][0], c[0][1][1]) 
            x3 = (c[0][2][0], c[0][2][1]) 
            x4 = (c[0][3][0], c[0][3][1])   
            im_dst = frame
            size = im_src.shape
            pts_dst = np.array([x1, x2, x3, x4])
            pts_src = np.array(
                           [
                            [0,0],
                            [size[1] - 1, 0],
                            [size[1] - 1, size[0] -1],
                            [0, size[0] - 1 ]
                            ],dtype=float
                           );
            
            h, status = cv2.findHomography(pts_src, pts_dst)
            temp = cv2.warpPerspective(im_src.copy(), h, (org_frame.shape[1], org_frame.shape[0])) 
            cv2.fillConvexPoly(org_frame, pts_dst.astype(int), 0, 16);
            org_frame = cv2.add(org_frame, temp)
        cv2.imshow('frame', org_frame)
    else:
        cv2.imshow('frame', frame)
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
