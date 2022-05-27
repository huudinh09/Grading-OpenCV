import cv2
import numpy as np


def rectContour(contours):

    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print("conner points",len(approx))
            if len(approx)==4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon

def getCornerPoints (Contour):
    peri = cv2.arcLength(Contour, True)
    approx = cv2.approxPolyDP(Contour, 0.02 * peri, True)
    return approx

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print("my points" ,myPoints)
    # print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)] #[0, 0]
    myPointsNew[3] = myPoints[np.argmax(add)] #[w, h]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] #[w, 0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[0, h]

    return myPointsNew

def spiltBoxes(img):

    rows = np.vsplit(img, 20)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 4)
        for box in cols :
            boxes.append(box)
            # cv2.imshow("Split", box)
    return boxes

def showAnswers(img, myIndex, grading, ans, questions, choices):
    # secW = int(img.shape[1]/questions)
    # secH = int(img.shape[0]/choices)
    secW = 110
    secH = 20
    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (secW + myAns*160)
        cY = (secH +x*35)
        cv2.circle(img, (cX, cY), 20, (0, 255, 0), cv2.FILLED)# 160, 35
        if grading[x] == 1:
            myColor = (0, 255, 0)
        else:
            myColor = (0, 0 , 255)
            correctAns = ans[x]
            cv2.circle(img, (secW + correctAns*160, secH + x*35), 20,
                       (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (cX, cY), 20, myColor, cv2.FILLED)# 160, 35

    return img
