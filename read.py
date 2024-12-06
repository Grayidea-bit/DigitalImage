import cv2 
global img 
def onMouse(event, x, y, flags, param):
    x,y = y,x
    print(x,y)
    print(img[x,y])
img = cv2.imread("name.jpg",0)
cv2.namedWindow("onMouse")
cv2.setMouseCallback("onMouse",onMouse)
cv2.imshow("onMouse",img)
cv2.waitKey()