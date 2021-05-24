import cv2
import numpy as np

frameWidth = 1000
frameHeight = 600
FILE_PATH = 'videos\\'

# put your video name here
FILE_NAME = 'IMG_9230.MP4'

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
cv2.createTrackbar("Threshold1", "Parameters", 99, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 99, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)

# Tran limits
cv2.createTrackbar("Tran low", "Parameters", 10000, 30000, empty)
cv2.createTrackbar("Tran med", "Parameters", 20000, 30000, empty)
cv2.createTrackbar("Tran high", "Parameters", 30000, 40000, empty)

# Long limits
cv2.createTrackbar("Long low", "Parameters", 10000, 30000, empty)
cv2.createTrackbar("Long med", "Parameters", 20000, 30000, empty)
cv2.createTrackbar("Long high", "Parameters", 30000, 40000, empty)

# Aleg limits
cv2.createTrackbar("Aleg low", "Parameters", 10000, 30000, empty)
cv2.createTrackbar("Aleg med", "Parameters", 20000, 30000, empty)
cv2.createTrackbar("Aleg high", "Parameters", 30000, 40000, empty)

# hole limits
cv2.createTrackbar("hole low", "Parameters", 10000, 50000, empty)
cv2.createTrackbar("hole med", "Parameters", 20000, 70000, empty)
cv2.createTrackbar("hole high", "Parameters", 30000, 80000, empty)


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


def get_severity(param, area):
    if area < cv2.getTrackbarPos(param + " low", "Parameters"):
        return 'low'
    elif area < cv2.getTrackbarPos(param + " med", "Parameters"):
        return 'Medium'
    else:
        return 'High'


def get_contours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Calculate Moments
    # moments = cv2.moments(img)
    # Calculate Hu Moments
    # huMoments = cv2.HuMoments(moments)
    # for i in range(0, 7):
    #     huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # areaMin = 15000
        areaMin = cv2.getTrackbarPos("Area", "Parameters")

        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            # print(peri)
            # x, y, w, h = cv2.boundingRect(approx)
            center, boundary, angle = cv2.minAreaRect(approx)
            w = int(boundary[0])
            h = int(boundary[1])
            x = int(center[0])
            y = int(center[1])
            rect = cv2.minAreaRect(approx)
            print(x)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            if w < 100 and h > w * 2:
                severity = get_severity("Long", area)
                crack_type = 'Longitudinal '

            elif w > h * 2 and h < 100:
                severity = get_severity("Tran", area)
                crack_type = 'Transverse '

            elif len(approx) > 5 and abs(h - w) < 200:
                severity = get_severity("Aleg", area)
                crack_type = 'Aleg'

            else:
                severity = get_severity("hole", area)
                crack_type = 'hole'

            cracks_results.append((severity, crack_type, area))

            cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 2)

            # cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + 20, y + 20), cv2.QT_FONT_NORMAL, 17,
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
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    # threshold1 = 255
    # threshold2 = 128
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((7, 7))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    get_contours(imgDil, imgContour)
    imgStack = stackImages(0.3, ([img, imgCanny],
                                 [imgDil, imgContour]))
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success, img = cap.read()

LOW_LONG = 0
MED_LONG = 0
HI_LONG = 0

Longitudinal = Crack(0, 0, 0)
Transverse = Crack(0, 0, 0)
Aleg = Crack(0, 0, 0)
Hole = Crack(0, 0, 0)

for crack in cracks_results:
    if crack[1] == 'Longitudinal' and crack[0] == 'low':
        Longitudinal.low += crack[2]
    elif crack[1] == 'Longitudinal' and crack[0] == 'Medium':
        Longitudinal.med += crack[2]
    elif crack[1] == 'Longitudinal' and crack[0] == 'High':
        Longitudinal.hi += crack[2]

    elif crack[1] == 'Transverse' and crack[0] == 'low':
        Transverse.low += crack[2]
    elif crack[1] == 'Transverse' and crack[0] == 'Medium':
        Transverse.med += crack[2]
    elif crack[1] == 'Transverse' and crack[0] == 'High':
        Transverse.hi += crack[2]

    elif crack[1] == 'Aleg' and crack[0] == 'low':
        Aleg.low += crack[2]
    elif crack[1] == 'Aleg' and crack[0] == 'Medium':
        Aleg.med += crack[2]
    elif crack[1] == 'Aleg' and crack[0] == 'High':
        Aleg.hi += crack[2]

    elif crack[1] == 'hole' and crack[0] == 'low':
        Hole.low += crack[2]
    elif crack[1] == 'hole' and crack[0] == 'Medium':
        Hole.med += crack[2]
    else:
        Hole.hi += crack[2]

LANE_WIDTH = 3

AC_INDEX = 100 - 40 * (Aleg.low / (0.02 * LANE_WIDTH * 70) + Aleg.med / (0.02 * LANE_WIDTH * 30) + Aleg.hi / (
        0.02 * LANE_WIDTH * 10))
TC_INDEX = 100 - 20 * (Transverse.low / (0.02 * LANE_WIDTH * 15.1) + Transverse.med / (
            0.02 * LANE_WIDTH * 7.5) + 40 * Transverse.hi / (
                               0.02 * LANE_WIDTH * 1.9))
LC_INDEX = 100 - 40 * (Longitudinal.low / (0.02 * LANE_WIDTH * 350) + Longitudinal.med / (
            0.02 * LANE_WIDTH * 200) + Longitudinal.hi / (
                               0.02 * LANE_WIDTH * 75))
PATCH_INDEX = 100 - 40 * (Hole.low / (0.02 * LANE_WIDTH * 160) + Hole.med / (0.02 * LANE_WIDTH * 80) + Hole.hi / (
        0.02 * LANE_WIDTH * 40))

SCR = 100 - (100 - AC_INDEX + 100 - LC_INDEX + 100 - TC_INDEX + 100 - PATCH_INDEX)
PCR = 0.6 * SCR + 40
