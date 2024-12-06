import cv2 as cv;
import numpy as np;

def makePicShow(name, img):
    cv.imshow(name,img);
    cv.imwrite(name+".jpg",img);
    cv.waitKey();
    
def makeHistogramArray(img):
    (x, y) = img.shape;
    pixelArray = np.zeros(256, dtype=int);

    for i in range(x):
        for j in range(y):
            pixelArray[img[i, j]] += 1

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

def makeEqualization(pixelArray,name):
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
    makePicShow(name,equalizationArray); 

    return cdf;
    

img = cv.imread("./pic/dog.jpg");  #讀入原圖
(x, y, z) = img.shape;
makePicShow("ori",img);  

blue = img[:, :, 0];
green = img[:, :, 1];
red = img[:, :, 2];

pixelArray = makeHistogramArray(blue);
blueHistogram = makeHistogramPictrue(pixelArray);
makePicShow("Blue",blueHistogram);  
blueCdf = makeEqualization(pixelArray,"Blueequalization");

pixelArray = makeHistogramArray(green);
greenHistogram = makeHistogramPictrue(pixelArray);
makePicShow("Green",greenHistogram);  
greenCdf = makeEqualization(pixelArray,"Greenequalization");


pixelArray = makeHistogramArray(red);
redHistogram = makeHistogramPictrue(pixelArray);
makePicShow("Red",redHistogram);  
redCdf = makeEqualization(pixelArray,"Redequalization");

equalization = np.zeros((x,y,z),np.uint8)
for k in range(z):
    for i in range(x):
        for j in range(y):
            if(k==0):
                equalization[i][j][k] = blueCdf[img[i][j][k]];
            if(k==1):
                equalization[i][j][k] = greenCdf[img[i][j][k]];
            if(k==2):
                equalization[i][j][k] = redCdf[img[i][j][k]];

makePicShow("equalization",equalization); 

