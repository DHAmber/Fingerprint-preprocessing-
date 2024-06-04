import cv2 as cv
import numpy as np
from numpy import asarray
import math
from PIL import Image

#reading an image
imag=cv.imread('1_3.tif',0)
cv.imshow('original fingerprint',imag)
IMG=np.asarray(imag)
print(IMG)
cv.waitKey(0)
cv.destroyAllWindows()
#segmementing teh image
blocksize=12
threshold=0.2
globalthreshold=np.std(imag)*threshold
segment_img=imag.copy()
img_zero=np.zeros_like(imag)
img_ones=np.ones_like(imag)
#cv.imshow('segmented fingerprint',segment_img)
numpydata = asarray(imag)
length,width=numpydata.shape
#print(x,y)
print(numpydata)
for i in range(0, width, blocksize):
    for j in range(0, length, blocksize):
        if (i + blocksize < width):
            x = i + blocksize
        else:
            x = width
        if (j + blocksize < length):
            y = j + blocksize
        else:
            y = length
        box = [i, j, x, y]
        # box=[10,5,15,18]
        img_ones[box[1]:box[3], box[0]:box[2]] = np.std(imag[box[1]:box[3], box[0]:box[2]])
        (img_ones[img_ones<globalthreshold])=0
            #img_ones=0
for i in range(0,width):
    for i in range(0,length):
        segment_img[i,j]=img_ones[i,j]*img_zero[i,j]
cv.imshow('segmented image',segment_img)
cv.waitKey(0)
cv.destroyAllWindows()

#normalization starts
m0=100
v0=100
norm_imag=imag.copy()
m=np.mean(imag)
v=np.std(imag)*2
for i in range(0,width):
    for i in range(0,length):
        if(norm_imag[i,j]>m):
            norm_imag[i,j]=m0+math.sqrt((((v0*norm_imag[i,j])-m)*2)/v)
        else:
            norm_imag[i,j] = abs(m0-math.sqrt(abs(((v0*norm_imag[i,j])-m)*2)/v))
cv.imshow('normalize image',norm_imag)
cv.waitKey(0)
cv.destroyAllWindows()

