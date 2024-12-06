import cv2 as cv;
import numpy as np;

# _____長_____
#|           |               
#|           寬
#|___________|


def con(img,m,n,kernel,bias,padding):
    if(padding==1): #補白邊 
        pad = np.zeros((m+(len(kernel[0])-1)*2,n+(len(kernel)-1)),np.uint8);        #創建一個空圖片(黑)
        pad.fill(255);                                                              #填滿白色
        pad[len(kernel[0])-1:len(kernel[0])-1+m,len(kernel)-1:len(kernel)-1+n]=img; #將原本的圖片放到中間
    elif(padding==0):   #補黑邊
        pad = np.zeros((m+(len(kernel[0])-1)*2,n+(len(kernel)-1)),np.uint8);
        pad[len(kernel[0])-1:len(kernel[0])-1+m,len(kernel)-1:len(kernel)-1+n]=img; 
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

def signName(img,name,m,n):
    (x1,y1)=name.shape;
    if(m>n): adjust = max(1,x1//(m//10));
    else: adjust = max(1,y1//(n//10));
    ad_name = pool(name,adjust,adjust,1);
    (x2,y2)=ad_name.shape;

    if (m - x2 >= 0) and (n - y2 >= 0):
        for j in range(y2):
            for i in range(x2):
                if(ad_name[i][j]>=245 or ad_name[i][j]<=200):
                    img[m-x2+i-10][n-y2+j-10]=0;
    return img;


img = cv.imread("self.jpg",0);  #讀入原圖
name = cv.imread("name.jpg",0);
cv.imshow("img",img);           #展示原圖
cv.imshow("name",name);
cv.waitKey();                   #等待任意鍵


average = [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]];  #average filter
sobel = [[-1,0,1],[-2,0,2],[-1,0,1]];                   #sobel filter
gaussian = [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]; #gaussian filter               

(x,y) = img.shape;                      #原圖長寬             
Out_con = con(img,x,y,average,0,-1);    #計算Con
temp = signName(Out_con,name,x,y);

cv.imshow("signed",temp);           #輸出Con後的圖片
cv.waitKey();
