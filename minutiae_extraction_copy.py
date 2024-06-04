import cv2 as cv
import numpy as np
import math
from skimage.morphology import skeletonize, thin
import glob as fileOpener
import fingerprint_enhancer

GLOBAL_THRESHOLD = 0.01
blockSize=16
def ShowImg(img,messsage):
    cv.imshow(messsage, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def _skeletonize(img):
    img = np.uint8(img > 128)
    _skel = skeletonize(img)
    _skel = np.uint8(_skel) * 255
    return _skel

def GetSegmentedImage(im):
    threshold = 0.2
    blocksize = 10
    segmented_image = im.copy()
    image_variance = np.zeros(im.shape)
    threshold = np.std(im)* threshold
    (length,width)=np.shape(im)
    mask=np.ones_like(im)
    for i in range(0,width,blocksize):
        for j in range(0,length,blocksize):
            if (i+blocksize<width):
                x=i+blocksize
            else:
                x=width
            if (j+blocksize<length):
                y=j+blocksize
            else:
                y=length
            box = [i, j, x, y]
            #box=[10,5,15,18]
            image_variance[box[1]:box[3], box[0]:box[2]]=np.std(im[box[1]:box[3], box[0]:box[2]])

    mask[image_variance < threshold] = 0
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (blocksize*2 , blocksize*2 ))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k)

    x, y = im.shape
    for i in range(0,x):
        for j in range(0,y):
            segmented_image[i,j]=segmented_image[i,j]*mask[i,j]
            #im.show(segmented_image)
    return segmented_image

def NormalizedImage(img):
    M=np.mean(img)
    V=np.std(img)**2
    result=np.copy(img)
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            a=math.sqrt((float(100) * ((img[i,j] - M) ** 2)) / V)
            if(img[i,j]<M):
                b= float(100)+a
            else:
                b= float(100)-a
            result[i,j]=b

    return result

def binarization(image):
    ret,thresh1= cv.threshold(image,120,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return thresh1
    #ShowImg("Binarized",ret)
    '''title=['binary']
    imageb=[thresh1]
    plt.subplot(2, 3, i + 1), plt.imshow(imageb, 'gray', vmin=0, vmax=255)
    plt.title(title)
    plt.xticks([]), plt.yticks([])/'''


def minutiae_at(pixels, i, j):
    #kernel_size=3
    # if middle pixel is black (represents ridge)
    if pixels[i][j] == 1:
        cells=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0)]
        values = [pixels[i + k][j + l] for k, l in cells]


        # count crossing how many times it goes from 0 to 1
        crossings = 0
        for k in range(0, len(values)-1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"
    return "none"


def calculate_minutiaes(im):
    binary_image = np.zeros_like(im)
    binary_image[im<10] = 1.0
    binary_image = binary_image.astype(np.int8)
    resultArray=[]
    (y, x) = im.shape
    # iterate each pixel minutia
    for i in range(1, y - 1):
        for j in range(1, x - 1):
            minutiae = minutiae_at(binary_image, i, j)
            if minutiae != "none":
                resultArray.append((j,i))
    return resultArray

def get_Minutiae(filePath):
    img_path = fileOpener.glob(filePath)
    img = np.array(cv.imread(img_path[0], 0))
    segmented=GetSegmentedImage(img)
    normalized=NormalizedImage(segmented)
    enhanced = fingerprint_enhancer.enhance_Fingerprint(normalized)
    binarizationImg = binarization(enhanced)
    skeleton = _skeletonize(binarizationImg)
    arr=calculate_minutiaes(skeleton)
    result=skeleton
    #result = cv.cvtColor(skeleton)
    radius = 1
    color = (255,255,0)
    thickness =1
    for minute in arr:
        cv.circle(result, (minute[0],minute[1]), radius, color, thickness)
    ShowImg(result,"Final Image")
    with open('Result_101_2.txt','a+') as f:
        for a in arr:
            f.write(str(a[0])+' '+str(a[1])+'\n')
minutaieArr=get_Minutiae('101_2.tif')

def false_minutiae_removal('Result_101_2.txt'):
 with open 'Result_101_2.txt' as f:
     f.readlines()













def removeFalseMinutaie(arr,count):
    dist_thrshold=10
    if count==999:
        return 'still'
    for e in arr:
        for other in arr:
            if e!=other:
                d=math.dist(e,other)
                if abs(d)<dist_thrshold:
                    arr.remove(other)
                    count+=1
                    removeFalseMinutaie(arr,count)
    return arr

def get_Minutiae(filePath):
    #img_path = fileOpener.glob(filePath)
    img = np.array(cv.imread(filePath, 0))
    ShowImg(img,"Original fingerprint")
    segmented=GetSegmentedImage(img)
    ShowImg(segmented, "Segmented fingerprint")
    normalized=NormalizedImage(segmented)
    ShowImg(normalized, "Normalized fingerprint")
    enhanced = fingerprint_enhancer.enhance_Fingerprint(normalized)
    binarizationImg = binarization(enhanced)
    ShowImg(binarizationImg,"Binarized Image")
    skeleton = _skeletonize(binarizationImg)
    ShowImg(skeleton, "Skeletonize fingerprint")
    arr=calculate_minutiaes(skeleton)
    with open('Result_101_2.txt','a+') as f:
        for a in arr:
            f.write(str(a[0])+' '+str(a[1])+'\n')
    print(arr)
minutaieArr=get_Minutiae('101_2.tif')