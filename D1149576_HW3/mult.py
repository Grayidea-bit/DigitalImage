import cv2 as cv;
import numpy as np;

def con(img,m,n,kernel,bias,padding):
    if(padding==1): #補白邊 
        pad = np.zeros((m+(len(kernel[0])-1)*2,n+(len(kernel)-1)),np.uint8);        #創建一個空圖片(黑)
        
        for j in range(n+(len(kernel)-1)):                                          #填滿白色
            for i in range(m+(len(kernel[0])-1)*2):
                pad[i][j] = 255;                           
        
        for j in range(n+(len(kernel)-1)):                                          #將原本的圖片放到中間
            for i in range(m+(len(kernel[0])-1)):
                if((i<len(kernel[0])-1+m and i>=len(kernel[0])-1) and (j<len(kernel)-1+n and j>=len(kernel)-1)):
                    pad[i][j] = img[i-(len(kernel[0])-1)][j-(len(kernel)-1)];                           
    elif(padding==0):   #補黑邊
        pad = np.zeros((m+(len(kernel[0])-1)*2,n+(len(kernel)-1)),np.uint8);
        
        for j in range(n+(len(kernel)-1)):                                          #將原本的圖片放到中間
            for i in range(m+(len(kernel[0])-1)):
                if((i<len(kernel[0])-1+m and i>=len(kernel[0])-1) and (j<len(kernel)-1+n and j>=len(kernel)-1)):
                    pad[i][j] = img[i-(len(kernel[0])-1)][j-(len(kernel)-1)];  
    else:
        pad = img;  #不補邊
    
    (x,y) = pad.shape;
    if(padding==1 or padding==0):   #如果有補邊則使用原圖片的長寬作為計算次數
        leng=m;
        wid=n;
    else:                           #若無則計算輸出圖片的大小作為計算次數
        leng=m-len(kernel[0])+1;
        wid=n-len(kernel)+1;
    
    Out = np.zeros((leng,wid),np.uint8);                                #創建輸出用的圖片
    for w_num in range(wid):                                            #寬的運行次數
        for l_num in range(leng):                                       #長的運行次數
            temp=0;                                                     #記錄點的數值
            for w in range(len(kernel)):                                #kernel的寬
                for l in range(len(kernel[w])):                         #kernel的長
                    temp += int(pad[l+l_num][w+w_num])*kernel[l][w];    #計算
            temp += bias;                                               #最後加上偏差值
            Out[l_num][w_num] = max(0, min(255, temp));                 #將範圍限制到0~255
    return Out;

def pool(img,size,stride,type):
    (x,y) = img.shape;                          #原圖片的長寬
    length = (x-size)//stride+1;                #計算縮小後的圖片長度
    width = (y-size)//stride+1;                 #計算縮小後的圖片寬度
    Out = np.zeros((length, width),np.uint8);   #建立輸出的圖片

    if(type==0):    #average pool
        for w_num in range(width):                                  #寬數量
            for l_num in range(length):                             #長數量
                temp=0;total=0;                                     #temp平均後給Out的值，total為加總
                for w in range(w_num*stride,w_num*stride+size):     #size要得寬度
                    for l in range(l_num*stride,l_num*stride+size): #size要的長度
                        if(w<y and l<x):                            #避免超出範圍
                            total += int(img[l][w]);                #加總
                temp = total//(size*size);                          #平均
                Out[l_num][w_num] = max(0, min(255, temp));         #將範圍限制到0~255
    else:   #max pool
        for w_num in range(width):  
            for l_num in range(length):
                Fmax=0;
                for w in range(w_num*stride,w_num*stride+size):
                    for l in range(l_num*stride,l_num*stride+size):
                        if(Fmax<img[l][w] and w<y and l<x):
                            Fmax = img[l][w];
                Out[l_num][w_num] = max(0, min(255, Fmax));
    return Out;

def convolution(img,kernel):
    (m,n)=img.shape;
    pad = np.zeros((m+2, n+2), dtype=np.uint8);        #創建一個空圖片(黑)

    for j in range(n):                                          #將原本的圖片放到中間
        for i in range(m):
            pad[i+1][j+1] = img[i][j];

    Out = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            temp=0;
            for w in range(len(kernel)):                       
                for l in range(len(kernel[w])):          
                    temp += int(pad[i+l][j+w])*kernel[l][w];
            Out[i][j] = max(0, min(255, temp)); 
    return Out;

def sobel(img):
    vertical = [[-1,0,1],[-2,0,2],[-1,0,1]];
    horizontal = [[-1,-2,-1],[0,0,0],[1,2,1]];

    Gx = convolution(img,horizontal);
    Gy = convolution(img,vertical);
    M = np.sqrt(Gx**2+Gy**2);
    return M;

def nonMaximum(img):
    Out = np.zeros_like(img);
    (x,y)=img.shape;
    for i in range(1,x-1):
        for j in range(1,y-1):
            now = img[i][j];
            #0  
            leftN = img[i-1][j];
            rightN = img[i+1][j];
            if(now<=leftN and now<=rightN):
                Out[i][j] = 0;
                continue;
            #45
            leftN = img[i-1][j-1];
            rightN = img[i+1][j+1];
            if(now<=leftN and now<=rightN):
                Out[i][j] = 0;
                continue;
            #90
            leftN = img[i][j-1];
            rightN = img[i][j+1];
            if(now<=leftN and now<=rightN):
                Out[i][j] = 0;
                continue;
            #135
            leftN = img[i-1][j+1];
            rightN = img[i+1][j-1];
            if(now<=leftN and now<=rightN):
                Out[i][j] = 0;
                continue;
            #finally
            Out[i][j]=now;
    return Out;

def thresholding(img):
    high = 250;
    low = 200;
    Out = np.zeros_like(img);
    (x,y)=img.shape;
    for i in range(x):
        for j in range(y):
            now = img[i][j];
            if(now>high): Out[i][j]=255;
            elif(now<low): Out[i][j]=0;
            else:
                for m in range(i-1,i+1):
                    for n in range(j-1,j+1):
                        if(m!=i and n!=i):
                            if(img[m][n]>high):
                                Out[i][j]=255;
                if(Out[i][j]!=255):
                    Out[i][j]=255;
    return Out;


def vote(img):
    max=0;
    (x,y)=img.shape;
    maxSize = int(np.sqrt(x**2+y**2));
    acc = np.zeros((maxSize*2,180),dtype=int);
    for i in range(x):
        for j in range(y):
            if(img[i][j]==255):
                for d in range(180):
                    r = d*3.14/180;
                    s = int(j*np.cos(r)+i*np.sin(r));
                    #print(s+maxSize,r);
                    acc[s+maxSize,d]+=1;
                    if(acc[s+maxSize,d]>max): max = acc[s+maxSize,d];
    return acc, max;

def findLine(img, acc, max):
    (x,y) = img.shape;
    maxSize = int(np.sqrt(x**2+y**2));
    threshold = max * 0.3;
    line=[];
    for i in range(maxSize*2):
        for j in range(180):
            #print(acc[i][j]);
            if(acc[i][j]>threshold):
                line.append([i-maxSize,j]);
    #print("Lines found:", line);
    print(threshold);
    return line;


def drawLine(img, line):
    Out = np.copy(img);
    height, width, c = img.shape
    for rho, theta in line:
        #print("Drawing line with rho={",rho,"}, theta={",theta,"}")  # 添加打印语句
        r = theta*3.14/180;
        a = np.cos(r)
        b = np.sin(r)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv.line(Out, (x1, y1), (x2, y2), (0,0,255), 1)
    return Out;

def signName(img,name,m,n):
    (x1,y1)=name.shape;                     #簽名檔的長寬
    if(m>n): adjust = max(1,x1//(m//10));   #長邊較大則縮成原圖長邊的1/10
    else: adjust = max(1,y1//(n//10));      #寬邊較大則縮成原圖寬邊的1/10
    tempName = sobel(name);
    tempName = nonMaximum(tempName);
    tempName = thresholding(tempName);
    ad_name = pool(tempName,adjust,adjust,1);   #利用pool縮小
    (x2,y2)=ad_name.shape;                  #縮小後的長寬

    #print(ad_name);
    if (m - x2 >= 0) and (n - y2 >= 0):     #確保範圍正確
        for j in range(y2):
            for i in range(x2):
                if(ad_name[i][j]>30): 
                    img[m-x2+i-10][n-y2+j-10][0]=0;
                    img[m-x2+i-10][n-y2+j-10][1]=255;
                    img[m-x2+i-10][n-y2+j-10][2]=0;
    return img;

gaussian = [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]; #gaussian filter               

ori = cv.imread("DigitalImage/D1149576_HW3/self.jpg",0);
rgb = cv.imread("DigitalImage/D1149576_HW3/self.jpg");
name = cv.imread("DigitalImage/D1149576_HW3/name.jpg",0);
(x,y)=ori.shape;
temp = con(ori,x,y,gaussian,0,0);
temp = sobel(temp);
temp = nonMaximum(temp);
temp = thresholding(temp);
cv.imwrite("DigitalImage/D1149576_HW3/thresholding3.jpg",temp);
acc,m = vote(temp);
line = findLine(temp,acc,m);
temp = drawLine(rgb,line);
temp = signName(temp,name,x,y);
cv.imwrite("DigitalImage/D1149576_HW3/finish3.jpg",temp);

