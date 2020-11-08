#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:08:49 2020

@author: kumar
"""

import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def v_compute(i,j,H):
    #print H
    v_row=np.zeros((1,6), np.float32)
    
    v_row[0][0]=H[0][i]*H[0][j]
    v_row[0][1]=H[0][i]*H[1][j]+H[1][i]*H[0][j]
    v_row[0][2]=H[1][i]*H[1][j]
    v_row[0][3]=H[2][i]*H[0][j]+H[0][i]*H[2][j]
    v_row[0][4]=H[2][i]*H[1][j]+H[1][i]*H[2][j]
    v_row[0][5]=H[2][i]*H[2][j]
    
    return v_row

"""def fn_compute(objpointsHomgenous,A0,A1,A2,A3,A4,A5,A6,A7,A8,RT0,RT1,RT2,RT3,RT4,RT5,RT6,RT7,RT8):
#def fn_compute(objpointsHomgenous,A,RT,k1,k2):    
    A=np.array([[A0,A1,A2],[A3,A4,A5],[A6,A7,A8]])
    RT=np.array([[RT0,RT1,RT2],[RT3,RT4,RT5],[RT6,RT7,RT8]])
    #A=A.reshape(3,3)
    #RT=RT.reshape(3,3)
    objpointsHomgenous=objpointsHomgenous.reshape(54,3)
    #H1=np.dot(A,RT)
    cameraFrameCoord=(np.dot(RT,objpointsHomgenous.T)).T
    normalized_coord=cameraFrameCoord/(cameraFrameCoord[:,2].reshape(54,1))
    
    normalizedSquareDist=np.square(normalized_coord[:,1])+np.square(normalized_coord[:,0])
    u0,v0=A2,A5
    #distortionFactor=k1*normalizedSquareDist+(k2*np.square(normalizedSquareDist))
    
    imgPointEst=np.dot(A,normalized_coord.T).T
    #print(objpointsHomgenous)
    imgPointEst=imgPointEst/(imgPointEst[:,2].reshape(54,1))
    #imgPointEst[:,0]=imgPointEst[:,0]+(imgPointEst[:,0]-u0)*distortionFactor
    #imgPointEst[:,1]=imgPointEst[:,1]+(imgPointEst[:,1]-v0)*distortionFactor
    #imgPointEstimated.append(imgPointEst)
        
    return imgPointEst.ravel()    """        
imgPointsStore=[]
def fn_compute(objpointsHomgenous,A0,A1,A2,A3,A4,A5,A6,A7,A8,k1,k2,*args):
#def fn_compute(objpointsHomgenous,A,RT,k1,k2):    
    AT=np.array([[A0,A1,A2],[A3,A4,A5],[A6,A7,A8]])
    RT=np.array(args[0:117]).reshape(13,3,3)
    #A=A.reshape(3,3)
    #print(len(kwargs))
    imgPointEstimated=[]
    objpointsHomgenous=objpointsHomgenous.reshape(13,54,3)
    #H1=np.dot(A,RT)
    for i in range(13):
        cameraFrameCoord=(np.dot(RT[i],objpointsHomgenous[i].T)).T
        normalized_coord=cameraFrameCoord/(cameraFrameCoord[:,2].reshape(54,1))
        
        normalizedSquareDist=np.square(normalized_coord[:,1])+np.square(normalized_coord[:,0])
        u0,v0=A2,A5
        distortionFactor=k1*normalizedSquareDist+(k2*np.square(normalizedSquareDist))
        
        imgPointEst=np.dot(AT,normalized_coord.T).T
        #print(objpointsHomgenous)
        imgPointEst=imgPointEst/(imgPointEst[:,2].reshape(54,1))
        imgPointEst[:,0]=imgPointEst[:,0]+(imgPointEst[:,0]-u0)*distortionFactor
        imgPointEst[:,1]=imgPointEst[:,1]+(imgPointEst[:,1]-v0)*distortionFactor
        imgPointEstimated.append(imgPointEst)
        imgPointsStore.append(imgPointEst)
    return (np.array(imgPointEstimated,dtype=np.float32)).ravel()    



path="./Calibration_Imgs/*.jpg"
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6,3), np.float32)
objpHomgenous = np.ones((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpHomgenous[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
objpointsHomgenous=[]#3d homogenous points with z neglected as z=0
imgpoints = [] # 2d points in image plane.
H=[]
V=np.zeros((26,6),np.float32)
for i, fil in enumerate(glob.glob(path)):
    image=cv2.imread(fil)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, corners=cv2.findChessboardCorners(gray,(9,6))
    if ret==True :
        objpoints.append(objp)
        objpointsHomgenous.append(objpHomgenous)
        
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners3=corners2.reshape(-1,2)
        corners3=np.column_stack((corners3,np.ones((54,1),np.float32)))
        imgpoints.append(corners3)
        h,_=cv2.findHomography(objp,imgpoints[i])
        #h=cv2.getPerspectiveTransform(objp,imgpoints[i])
        
        V[2*i]=v_compute(0,1,h)
        V[2*i+1]=v_compute(0,0,h)-v_compute(1,1,h)
        H.append(h)
        #img=cv2.drawChessboardCorners(image,(9,6),np.array(corners3[:,0:2].reshape(54,1,2),dtype=np.float32),False)
        #plt.figure(14)
        #plt.imshow(img)
        
eigval,eigvec=np.linalg.eig(np.dot(V.T,V))     
min_eigval_col=(np.argwhere(eigval==min(eigval))).item()
B=eigvec[:,min_eigval_col]


b11=B[0]
b12=B[1]
b22=B[2]
b13=B[3]
b23=B[4]
b33=B[5]

A=np.zeros((3,3),np.float32)
A[1][2]=((b12*b13)-(b11*b23))/((b11*b22)-np.power(b12,2))
l=b33-((np.power(b13,2)+(A[1][2]*((b12*b13)-(b11*b23))))/b11)
A[0][0]=np.sqrt(l/b11)
A[1][1]=np.sqrt((l*b11)/((b11*b22)-np.power(b12,2)))
A[0][1]=-(b12*(np.power(A[0][0],2))*A[1][1])/l
A[0][2]=(A[1][2]*A[0][1]/A[1][1])-(b13*np.square(A[0][0])/l)
A[2][2]=1.0

R=[]
T=[]
RT=[]
   
for i, hom in enumerate(H):
    l=1/(np.linalg.norm(np.dot(np.linalg.inv(A),hom[:,0])))
    r1=l*(np.dot(np.linalg.inv(A),hom[:,0]))
    r2=l*(np.dot(np.linalg.inv(A),hom[:,1]))
    r3=np.cross(r1,r2)
    R.append(np.column_stack((r1,r2,r3)))
    T.append(l*(np.dot(np.linalg.inv(A),hom[:,2])))
    RT.append(np.array(np.column_stack((r1,r2,l*(np.dot(np.linalg.inv(A),hom[:,2])))),dtype=np.float32))
    
#imgPointEstimated=fn_compute(objpointsHomgenous, A,RT)      
#imgPointEstimated=[]
k1=np.float32(0.0)
k2=np.float32(0.0)
"""for i in range (13):
    p0=A.ravel(),RT[i].ravel()
    p0=np.hstack(p0)
    coeff,x=curve_fit(fn_compute,objpointsHomgenous[i].ravel(),imgpoints[i].ravel(),p0) 
    #print(coeff)
    #A=coeff[0:9].reshape(3,3)
    #k1=coeff[18]
    #k2=coeff[19]

A_estimated=coeff[0:9].reshape(3,3)
print(A_estimated)"""


p0=A.ravel(),k1,k2,np.array(RT).ravel()
p0=np.hstack(p0)
objpointsHomgenous=np.array(objpointsHomgenous,dtype=np.float32)
imgpoints=np.array(imgpoints,dtype=np.float32)
coeff,x=curve_fit(fn_compute,objpointsHomgenous.ravel(),imgpoints.ravel(),p0) 
    #print(coeff)
    #A=coeff[0:9].reshape(3,3)
    #k1=coeff[18]
    #k2=coeff[19]

A_estimated=coeff[0:9].reshape(3,3)
k_estimated=np.zeros((1,4),dtype=np.float32)
k_estimated[0][0]=coeff[9]
k_estimated[0][1]=coeff[10]
RT_estimated=coeff[11:128].reshape(13,3,3)
print("Initial Intrinsic Matrix", A)
print("Estimated Intrinsic Matrix",A_estimated)
print("Estimated Radial distortion coefficients", k_estimated)

for i, fil in enumerate(glob.glob(path)):
    image=cv2.imread(fil)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    undistort_img=cv2.undistort(image,A,(k_estimated[0][0].item(),k_estimated[0][1].item(),0,0),A_estimated)
    undistort_img=cv2.drawChessboardCorners(undistort_img,(9,6),np.array(imgPointsStore[i][:,0:2].reshape(54,1,2),dtype=np.float32),False)
    fig=plt.figure(i+1,(1,2))
    fig.add_subplot(1,2,1)
    fig.suptitle("Left: distorted image and Right:undistorted image")
    image=cv2.drawChessboardCorners(image,(9,6),np.array(imgPointsStore[i][:,0:2].reshape(54,1,2),dtype=np.float32),False)
    plt.imshow(image)
    fig.add_subplot(1,2,2)
    plt.imshow(undistort_img)

tot_error = 0
R_estimated=[]
T_estimated=[]
for i in range(13):
    R_estimated.append(np.column_stack((RT_estimated[i][:,0],RT_estimated[i][:,1],np.cross(RT_estimated[i][:,0],RT_estimated[i][:,1]))))
    T_estimated.append(RT_estimated[i][:,2])
    rvec=cv2.Rodrigues(R_estimated[i])
    imgpoints2, _ = cv2.projectPoints(objpointsHomgenous[i], rvec[0], T_estimated[i], A_estimated, k_estimated)
    imgpoints2=np.array(imgpoints2.reshape(54,2),dtype=np.float32)
    #print(imgpoints2.shape)
    #print(imgpoints[i].shape)
    error = cv2.norm(imgpoints[i][:,0:2],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "Reprojection error: ", tot_error/13.0