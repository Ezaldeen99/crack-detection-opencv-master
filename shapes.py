import cv2
import numpy as np

# used only at once
found_result = False

IMG_PATH = 'photo5310294000407785994.jpg'

def empty(a):
    pass


# this is the default value for threshold 1
DEFAULT_AREA_MIN = 8000


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


def get_contours(img, imgContour, useTracker):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Calculate Moments
    # moments = cv2.moments(img)
    # Calculate Hu Moments
    # huMoments = cv2.HuMoments(moments)
    # for i in range(0, 7):
    #     huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    #
    # print(huMoments)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # areaMin = 15000
        areaMin = DEFAULT_AREA_MIN

        if useTracker:
            areaMin = cv2.getTrackbarPos("Area", "Parameters")

        if area > areaMin:

            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            # print(peri)
            # x, y, w, h = cv2.boundingRect(approx)
            center, boundary, angle = cv2.minAreaRect(approx)
            # crack width
            w = int(boundary[0])
            # crack height
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

            aspect_ratio = min(w, h) / max(w, h)

            if w < 100 and h > w * 2:
                severity = get_severity("Long", area)
                crack_type = 'Longitudinal '
            elif w > h * 2 and h < 100:
                severity = get_severity("Tran", area)
                crack_type = 'Transverse '
            elif len(approx) > 5:
                severity = get_severity("Aleg", area)
                crack_type = 'Aleg'
            else:
                severity = get_severity("hole", area)
                crack_type = 'hole'

            cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 2)

            # cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 4)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 4)

            cv2.putText(imgContour, crack_type, (x + 20, y + 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 4)

            cv2.putText(imgContour, severity, (x + 20, y + 140), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 4)

            return True


# this is the default value for threshold 1
DEFAULT_TH1 = 159

# this is the default value for threshold 2
DEFAULT_TH2 = 99

img = cv2.imread(IMG_PATH)
imgContour = img.copy()
imgBlur = cv2.GaussianBlur(img, (11, 11), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

for threshold1 in range(255, 90, -2):
    if found_result:
        break
# for threshold1 in range(90, 230, 2):
#     if found_result:
#         break
    for threshold2 in range(95, 255, 2):
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        found_result = get_contours(imgDil, imgContour, False)
        if found_result:
            DEFAULT_TH1 = threshold1
            DEFAULT_TH2 = threshold2
            break

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 900, 600)
cv2.createTrackbar("Threshold1", "Parameters", DEFAULT_TH1, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", DEFAULT_TH2, 255, empty)
cv2.createTrackbar("Area", "Parameters", DEFAULT_AREA_MIN, 30000, empty)

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

while True:
    img = cv2.imread(IMG_PATH)
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (11, 11), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    get_contours(imgDil, imgContour, True)
    imgStack = stackImages(0.3, ([img, imgCanny],
                                 [imgDil, imgContour]))
    cv2.imshow("original image", cv2.resize(img, (960, 540)))
    cv2.imshow("imgCanny", cv2.resize(imgCanny, (960, 540)))
    cv2.imshow("imgDil", cv2.resize(imgDil, (960, 540)))
    cv2.imshow("Result", cv2.resize(imgContour, (960, 540)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
