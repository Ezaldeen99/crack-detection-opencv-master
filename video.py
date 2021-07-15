## @author : ezzulddin sahib --2021



import cv2
import numpy as np

frameWidth = 1000
frameHeight = 600
FILE_PATH = 'videos\\'

# put your video name here
FILE_NAME = 'IMG_9229.MP4'

cap = cv2.VideoCapture(FILE_PATH + FILE_NAME)

cap.set(3, frameWidth)
cap.set(4, frameHeight)

# cracks results
cracks_results = []


def empty(a):
    pass


class Crack:
    def __init__(self, low, med, hi):
        self.low = low
        self.med = med
        self.hi = hi


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 900, 600)
cv2.createTrackbar("Threshold1", "Parameters", 70, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 140, 255, empty)
cv2.createTrackbar("Area", "Parameters", 4, 10, empty)

# Tran limits
cv2.createTrackbar("Tran low", "Parameters", 5, 10, empty)
cv2.createTrackbar("Tran med", "Parameters", 7, 10, empty)
cv2.createTrackbar("Tran high", "Parameters", 10, 10, empty)

# Long limits
cv2.createTrackbar("Long low", "Parameters", 5, 10, empty)
cv2.createTrackbar("Long med", "Parameters", 7, 10, empty)
cv2.createTrackbar("Long high", "Parameters", 10, 10, empty)

# Aleg limits
cv2.createTrackbar("Aleg low", "Parameters", 5, 10, empty)
cv2.createTrackbar("Aleg med", "Parameters", 7, 10, empty)
cv2.createTrackbar("Aleg high", "Parameters", 10, 10, empty)

# hole limits
cv2.createTrackbar("hole low", "Parameters", 5, 10, empty)
cv2.createTrackbar("hole med", "Parameters", 7, 10, empty)
cv2.createTrackbar("hole high", "Parameters", 10, 10, empty)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def get_severity(param, area, total_area):
    if area < (cv2.getTrackbarPos(param + " low", "Parameters") * total_area / 1000):
        return 'low'
    elif area < (cv2.getTrackbarPos(param + " med", "Parameters") * total_area / 1000):
        return 'Medium'
    else:
        return 'High'


def get_contours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # find the width and height of the image
    height, width, channels = imgContour.shape

    # image area
    total_area = height * width

    for cnt in contours:
        area = cv2.contourArea(cnt)

        area_min = cv2.getTrackbarPos("Area", "Parameters") * total_area / 1000

        if area > area_min:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            center, boundary, angle = cv2.minAreaRect(approx)

            w = int(boundary[0])
            h = int(boundary[1])
            x = int(center[0])
            y = int(center[1])
            rect = cv2.minAreaRect(approx)

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if abs(angle) > 45:
                temp = h
                h = w
                w = temp

            if w < 100 and h > w * 2:
                severity = get_severity("Long", area, total_area)
                crack_type = 'Longitudinal'

            elif w > h * 2 and h < 100:
                severity = get_severity("Tran", area, total_area)
                crack_type = 'Transverse'

            elif cv2.isContourConvex(approx):
                severity = get_severity("hole", area, total_area)
                crack_type = 'hole'

            else:
                print(len(approx))
                severity = get_severity("Aleg", area, total_area)
                crack_type = 'Aleg'

            cracks_results.append((severity, crack_type, area))

            # cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 2)

            # cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + 20, y + 20), cv2.QT_FONT_NORMAL, 1,
                        (0, 0, 0), 4)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 4)

            cv2.putText(imgContour, crack_type, (x + 20, y + 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 4)

            cv2.putText(imgContour, severity, (x + 20, y + 140), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 4)


success, img = cap.read()
while success:

    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (21, 21), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((7, 7))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    get_contours(imgDil, imgContour)
    imgStack = stackImages(0.5, ([img, imgCanny],
                                 [imgDil, imgContour]))
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success, img = cap.read()

Longitudinal = Crack(0, 0, 0)
Transverse = Crack(0, 0, 0)
Aleg = Crack(0, 0, 0)
Hole = Crack(0, 0, 0)

for crack in cracks_results:
    if crack[1] == 'Longitudinal' and crack[0] == 'low':
        Longitudinal.low += crack[2]
    elif crack[1] == 'Longitudinal' and crack[0] == 'Medium':
        Longitudinal.med += crack[2]
    elif crack[1] == 'Longitudinal':
        Longitudinal.hi += crack[2]

    elif crack[1] == 'Transverse' and crack[0] == 'low':
        Transverse.low += crack[2]
    elif crack[1] == 'Transverse' and crack[0] == 'Medium':
        Transverse.med += crack[2]
    elif crack[1] == 'Transverse':
        Transverse.hi += crack[2]

    elif crack[1] == 'Aleg' and crack[0] == 'low':
        Aleg.low += crack[2]
    elif crack[1] == 'Aleg' and crack[0] == 'Medium':
        Aleg.med += crack[2]
    elif crack[1] == 'Aleg':
        Aleg.hi += crack[2]

    elif crack[1] == 'hole' and crack[0] == 'low':
        Hole.low += crack[2]
    elif crack[1] == 'hole' and crack[0] == 'Medium':
        Hole.med += crack[2]
    else:
        Hole.hi += crack[2]

LANE_WIDTH = 3
# assume
# 1 PX = 0.00086805544619423 ft
ft = 0.0000000086805544619423
AC_INDEX = 100 - 40 * (
        Aleg.low * ft / (0.02 * LANE_WIDTH * 70) + Aleg.med * ft / (0.02 * LANE_WIDTH * 30) + Aleg.hi * ft / (
        0.02 * LANE_WIDTH * 10))

TC_INDEX = 100 - 20 * (Transverse.low * ft / (LANE_WIDTH * 15.1) + Transverse.med * ft / (
        LANE_WIDTH * 7.5) + 40 * Transverse.hi * ft / (
                               LANE_WIDTH * 1.9))

LC_INDEX = 100 - 40 * (Longitudinal.low * ft / (0.02 * 350) + Longitudinal.med * ft / (
        0.02 * 200) + Longitudinal.hi * ft / (
                               0.02 * 75))

PATCH_INDEX = 100 - 40 * (Hole.low + Hole.med + Hole.low) * ft / (0.02 * LANE_WIDTH * 80)

SCR = 100 - (100 - AC_INDEX + 100 - LC_INDEX + 100 - TC_INDEX + 100 - PATCH_INDEX)
PCR = 0.6 * SCR + 40

print(PCR)
if PCR <= 60:
    print('poor')
elif PCR <= 84:
    print('FAIR')
elif PCR <= 94:
    print('FAIR')
else:
    print('EXCELLENT')
