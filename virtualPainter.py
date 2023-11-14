import cv2
import numpy as np
import time
import os
import HandTracking as ht

##############
brushThickness = 15
eraserThickness = 50
##############

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = [cv2.imread(os.path.join(folderPath, imPath)) for imPath in myList]
print(len(overlayList))

header = overlayList[0]
drawColor = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1200, 3), np.uint8)

while True:
    # 1. Import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # Tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If both index and middle fingers are up - selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")
            if 250 < x1 < 450:
                header = overlayList[0]
                drawColor = (255, 0, 0)
            elif 500 < x1 < 650:
                header = overlayList[1]
                drawColor = (0, 165, 255)
            elif 700 < x1 < 850:
                header = overlayList[2]
                drawColor = (203, 192, 255)
            elif 950 < x1 < 1200:
                header = overlayList[3]
                drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If only the index finger is up - drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), brushThickness, drawColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # Resize imgInv to match the size of img
    imgInv = cv2.resize(imgInv, (img.shape[1], img.shape[0]))

    # Perform bitwise AND operation
    img = cv2.bitwise_and(img, imgInv)

    # Resize imgCanvas to match the size of img
    imgCanvas = cv2.resize(imgCanvas, (img.shape[1], img.shape[0]))

    # Perform bitwise OR operation
    img = cv2.bitwise_or(img, imgCanvas)

    # Set the header image
    img[0:128, 0:1280] = header

    # Resize the imgCanvas to match the size of img
    imgCanvas = cv2.resize(imgCanvas, (img.shape[1], img.shape[0]))

    # Combine the images using addWeighted
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit the loop
        break

cap.release()
cv2.destroyAllWindows()