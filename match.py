# -*- coding: utf-8 -*-
import cv2
import numpy as np

#匹配图像
def mathImg(smallImg,normalImg):
    smallImg = cv2.cvtColor(smallImg,cv2.COLOR_BGR2GRAY)
    M = np.ones(smallImg.shape,dtype='uint8')*75
    smallImg = cv2.subtract(smallImg,M)
    normalImg = cv2.cvtColor(normalImg,cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(smallImg, normalImg, cv2.TM_CCOEFF_NORMED)
    x, y = np.unravel_index(result.argmax(), result.shape)
    return [ x,y,x+smallImg.shape[1],y+smallImg.shape[0] ]

if __name__ == "__main__":
    smallImg = cv2.imread('./small.jpg')
    normalImg = cv2.imread('./normal.jpg')
    rect = mathImg(smallImg,normalImg)
    cv2.rectangle(normalImg,(rect[1],rect[0]),(rect[3],rect[2]),(0,0,0),3)
    cv2.imwrite('./result.jpg',normalImg)
    cv2.imshow('compare',normalImg)
    cv2.waitKey()
