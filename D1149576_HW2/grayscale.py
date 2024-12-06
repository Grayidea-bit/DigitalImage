import cv2 as cv;
import numpy as np;

import matplotlib.pyplot as plt

def makePicShow(name, img):
    cv.imshow(name,img);
    cv.imwrite(name+"gray.jpg",img);
    cv.waitKey();
    
def makeHistogramArray(img):
    (x,y) = img.shape;
    pixelArray = np.zeros(256, dtype=int);

    for i in img:
        pixelArray[i]+=1;

    return pixelArray;

def findMax(pixelArray):
    Max = pixelArray[0];

    for i in pixelArray:
        if(i>Max):
            Max = i;

    return Max;

def makeHistogramPictrue(pixelArray):
    count=0;
    Max = findMax(pixelArray);
    pic = np.zeros((Max,256),np.uint8);

    for i in range(Max):
        for j in range(256):
            pic[i][j] = 255; 

    for i in pixelArray:
        for j in range(i):
            pic[Max-j-1][count]=0;
        count+=1;


    return pic;

def makeEqualization(pixelArray):
    total=0;
    temp=0;
    Max = findMax(pixelArray);
    newArray = np.zeros(256, dtype=int);
    cdf = np.zeros(256, dtype=int);

    for i in pixelArray:
        total+=i;

    for i in range(len(pixelArray)):
        temp+=pixelArray[i];
        pos = temp*255//total;
        newArray[pos]=pixelArray[i];
        cdf[i]=pos;

    equalizationArray = makeHistogramPictrue(newArray);
    makePicShow("equalizationArray",equalizationArray); 

    return cdf;



    

img = cv.imread("./pic/123.jpg",0);  #讀入原圖
(x, y) = img.shape;
pixelArray = makeHistogramArray(img);
makePicShow("Ori",img);  

histogram = makeHistogramPictrue(pixelArray);
makePicShow("histogram",histogram);  

# hist = cv.calcHist([img], [0], None, [256], [0,256]);
# plt.bar(range(256), hist.flatten())
# plt.show()

cdf = makeEqualization(pixelArray);
equalization = np.zeros((x,y),np.uint8)
for i in range(x):
    for j in range(y):
        equalization[i][j] = cdf[img[i][j]];

makePicShow("equalization",equalization); 

