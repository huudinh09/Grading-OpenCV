







import cv2
import numpy as np
import ulits
import pytesseract
import pandas as pd
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()
answerPath = filedialog.askopenfilename()

###############################
answerFile = pd.read_csv(answerPath)
answerL = (answerFile.iloc[0:20, 1])
answerR = (answerFile.iloc[20:40, 1])


replaceAnsL = answerL.replace(['A', 'B', 'C', 'D'], [0, 1, 2, 3])
replaceAnsR = answerR.replace(['A', 'B', 'C', 'D'], [0, 1, 2, 3])


pytesseract.pytesseract.tesseract_cmd = 'E:\\OpenCV\\tesseract\\tesseract.exe'
path = "bai2.png"
widthImg = 700
heigthImg = 700
questions = 40
choices = 4


ans1 = replaceAnsL.to_numpy()
ans2 = replaceAnsR.to_numpy()

###############################



img = cv2.imread(path)



img= cv2.resize(img,(widthImg, heigthImg))
imgContours = img.copy()
imgFinal = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 5, 50)

#####FIND ALL CONTOURS############

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

#####FIND LEFT BIG RECTANGLES#######
rectCon = ulits.rectContour(contours)
leftBigCon = ulits.getCornerPoints(rectCon[0])

# print(biggestCon)
#####FIND RIGHT BIG RECTANGLES#######
rightBigCon = ulits.getCornerPoints(rectCon[1])

#####FIND GRADEPOINT###########
gradePoint = ulits.getCornerPoints(rectCon[5])
# print(gradePoint)

#####FIND NAME BOX###########
nameBox = ulits.getCornerPoints(rectCon[6])
# print(nameBox)
# cv2.rectangle(img, (162, 95), (340, 119),(255,0,0), 1 )

#####FIND ID BOX###########
idBox = ulits.getCornerPoints(rectCon[6])
# cv2.rectangle(imgFinal, (184,93), (365, 121),(255,0,0), 1 )


if leftBigCon.size != 0 and gradePoint.size != 0 and rightBigCon.size !=0:
    cv2.drawContours(imgContours, leftBigCon, -1, (255, 0, 0), 10)
    cv2.drawContours(imgContours, gradePoint, -1, (0, 0, 255), 10)
    cv2.drawContours(imgContours, rightBigCon, -1, (0, 0, 255), 10)
    cv2.drawContours(imgContours, nameBox, -1, (0, 0, 255), 10)
    cv2.drawContours(imgContours, idBox, -1, (0, 0, 255), 10)


    ###REODER NEW POINT#######

    leftBigCon = ulits.reorder(leftBigCon)
    gradePoint = ulits.reorder(gradePoint)
    rightBigCon = ulits.reorder(rightBigCon)
    nameBox = ulits.reorder(nameBox)
    idBox = ulits.reorder(idBox)

    ######WRAP PERSPECTIVE#######
    ## LEFT BIG CONTOUR
    pointL1 = np.float32(leftBigCon)
    pointL2 = np.float32([[0, 0], [widthImg, 0], [0, heigthImg], [widthImg, heigthImg]])
    matrixL1 = cv2.getPerspectiveTransform(pointL1, pointL2)
    imgWarpcoloredL = cv2.warpPerspective(img, matrixL1, (widthImg, heigthImg))

    ## RIGHT BIG CONTOUR
    pointR1 = np.float32(rightBigCon)
    pointR2 = np.float32([[0, 0], [widthImg, 0], [0, heigthImg], [widthImg, heigthImg]])
    matrixR1 = cv2.getPerspectiveTransform(pointR1, pointR2)
    imgWarpcoloredR = cv2.warpPerspective(img, matrixR1, (widthImg, heigthImg))

    ## GRADE POINT CONTOUR
    pointG1 = np.float32(gradePoint)
    pointG2 = np.float32([[0, 0], [300, 0], [0, 150], [300, 150]])
    matrixG = cv2.getPerspectiveTransform(pointG1, pointG2)
    imgWarpgrade = cv2.warpPerspective(img, matrixG, (300, 150))

    ## NAME BOX
    pointN1 = np.float32(nameBox)
    pointN2 = np.float32([[0, 0], [400, 0], [0, 60], [400, 60]])
    matrixN = cv2.getPerspectiveTransform(pointN1, pointN2)
    imgWarpName = cv2.warpPerspective(img, matrixN, (400, 60))
    # print(pytesseract.image_to_string(imgWarpName))

    ## ID BOX

    pointID1 = np.float32(idBox)
    pointID2 = np.float32([[0, 0], [400, 0], [0, 60], [400, 60]])
    matrixID = cv2.getPerspectiveTransform(pointID1, pointID2)
    imgWarpID = cv2.warpPerspective(img, matrixID, (400, 60))
    mssv = str(pytesseract.image_to_string(imgWarpID)).strip()
    mssv1 = mssv + ".jpg"
    print(mssv1)
    #####APPLY THRESHOLD FOR LEFT BIG CONTOUR#######
    imgWardGrayL = cv2.cvtColor(imgWarpcoloredL, cv2.COLOR_BGR2GRAY)
    imgThreshL = cv2.threshold(imgWardGrayL, 150, 255, cv2.THRESH_BINARY_INV)[1]

    #####APPLY THRESHOLD FOR RIGHT BIG CONTOUR#######
    imgWardGrayR = cv2.cvtColor(imgWarpcoloredR, cv2.COLOR_BGR2GRAY)
    imgThreshR = cv2.threshold(imgWardGrayR, 150, 255, cv2.THRESH_BINARY_INV)[1]

    boxesL = ulits.spiltBoxes(imgThreshL)
    boxesR = ulits.spiltBoxes(imgThreshR)




    ###PROCESS FOR LEFT BIG CONTOUR###
    myPixelValL = np.zeros((int(questions/2), choices))
    countCOfL = 0
    countROfL = 0
    for box in boxesL:
        totalPixelsL = cv2.countNonZero(box)
        myPixelValL[countROfL][countCOfL] = totalPixelsL
        countCOfL += 1
        if(countCOfL == choices):


            countROfL += 1
            countCOfL = 0

    ######FIND INDEX OF MARKED BOX##########
    myIndexL = []
    for x in range(0, int(questions/2)):
        arr = myPixelValL[x]
        myIndexValL = np.where(arr == np.amax(arr))
        myIndexL.append(myIndexValL[0][0])

    print(myIndexL)
    #GRADING

    gradingL = []
    for x in range(0, int(questions/2)):
        if(ans1[x] == myIndexL[x]):
            gradingL.append(1)
        else: gradingL.append(0)
    scoreL = sum(gradingL)/(questions/2) * 5

    ####PROCESS FOR RIGHT BIG CONTOUR
    countCOfR = 0
    countROfR = 0
    gradingR = []
    myPixelValR = np.zeros((int(questions/2), choices))
    for box in boxesR:
        totalPixelsR = cv2.countNonZero(box)
        myPixelValR[countROfR][countCOfR] = totalPixelsR
        countCOfR += 1
        if(countCOfR == choices): countROfR += 1; countCOfR = 0

    ######FIND INDEX OF MARKED BOX##########
    myIndexR = []
    for x in range(0, int(questions/2)):
        arr = myPixelValR[x]
        myIndexValR = np.where(arr == np.amax(arr))
        myIndexR.append(myIndexValR[0][0])
    print(myIndexR)
    #GRADING

    for x in range(0, int(questions/2)):
        if( ans2[x] == myIndexL[x]):
            gradingR.append(1)
        else: gradingR.append(0)
    scoreR = sum(gradingR)/(questions/2) * 5

    score = scoreL + scoreR
    # grading = gradingL + gradingR
    # myIndex = myIndexL + myIndexR
    # ans = ans1 + ans2
    #######DISPLAY ANSWERS######
    imgResultL = imgWarpcoloredL.copy()

    imgResultL = ulits.showAnswers(imgWarpcoloredL, myIndexL,
                                   gradingL, ans1, int(questions/2), choices)
    imgRawDrawingL = np.zeros_like(imgWarpcoloredL)

    imgRawDrawingL = ulits.showAnswers(imgRawDrawingL, myIndexL,
                                       gradingL, ans1, int(questions/2), choices)
    invMatrixL = cv2.getPerspectiveTransform(pointL2, pointL1)

    imgInvWarpL = cv2.warpPerspective(imgRawDrawingL, invMatrixL,
                                      (widthImg, heigthImg))

    imgResultR = imgWarpcoloredR.copy()
    imgResultR = ulits.showAnswers(imgWarpcoloredR, myIndexR, gradingR, ans2, int(questions/2), choices)
    imgRawDrawingR = np.zeros_like(imgWarpcoloredR)
    imgRawDrawingR = ulits.showAnswers(imgRawDrawingR, myIndexR, gradingR, ans2, int(questions/2), choices)
    invMatrixR = cv2.getPerspectiveTransform(pointR2, pointR1)
    imgInvWarpR = cv2.warpPerspective(imgRawDrawingR, invMatrixR, (widthImg, heigthImg))

#######DISPLAY SCORE
imgRawGrade = np.zeros_like(imgWarpgrade)
cv2.putText(imgRawGrade, str(float(score)), (80, 130), cv2.FONT_ITALIC, 3, (0, 255, 255), 5)
invMatrixG = cv2.getPerspectiveTransform(pointG2, pointG1)
imgInvWarpGrade = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heigthImg))


imgFinal = cv2.addWeighted(imgFinal, 0.9, imgInvWarpR, 1, 0)
imgFinal = cv2.addWeighted(imgFinal, 0.9, imgInvWarpL, 1, 0)
imgFinal = cv2.addWeighted(imgFinal, 0.9, imgInvWarpGrade, 1, 0)


#######DISPLAY##########
# cv2.imshow("grade", imgWarpgrade)
# cv2.imshow("abc", imgWarpcolored)
# cv2.imshow("box", boxes[0])
# cv2.imshow("Thresh hold", imgThresh)
# cv2.imshow("Test", img)
# cv2.imshow("Result", img)
cv2.imshow("ss", imgFinal)
cv2.waitKey(0)

cv2.imwrite("./Picture/" + mssv1, imgFinal)
