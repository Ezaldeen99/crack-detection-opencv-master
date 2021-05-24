import cv2
import numpy as np
import scipy.ndimage


def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi / 4)) % 4
    winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag


def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max


def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


# Load image, grayscale, median blur, Otsus threshold
# img = cv2.imread('photo_2021-04-16_13-23-34.jpg')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, 11)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# # Morph open
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
# #
# # # Find contours and filter using contour area and aspect ratio
# # cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
#
# # _, thrash = cv2.threshold(gray, 240, 255, cv2.CHAIN_APPROX_NONE)
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.imshow('shapes', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
#     area = cv2.contourArea(contour)
#     if len(approx) > 5 and area > 1000:
#         cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
#
#         x = approx.ravel()[0]
#         y = approx.ravel()[1] - 5
#         if len(approx) == 3:
#             cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#         elif len(approx) == 4:
#             x, y, w, h = cv2.boundingRect(approx)
#             aspectRatio = float(w) / h
#             print(aspectRatio)
#             if 0.95 <= aspectRatio < 1.05:
#                 cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#             else:
#                 cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#         elif len(approx) == 5:
#             cv2.putText(img, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#         elif len(approx) == 10:
#             cv2.putText(img, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#         else:
#             cv2.putText(img, "hole", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#
# cv2.imshow('shapes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
#     area = cv2.contourArea(c)
#     if len(approx) > 5 and area > 1000:
#         ((x, y), r) = cv2.minEnclosingCircle(c)
#         cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
#
# cv2.imshow('thresh', thresh)
# cv2.imshow('opening', opening)
# cv2.imshow('image', image)
# cv2.waitKey()

# start calulcation

image = cv2.imread('photo_2021-04-16_13-23-34.jpg')

# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


with_nmsup = True  # apply non-maximal suppression
fudgefactor = 0.8  # with this threshold you can play a little bit
sigma = 21  # for Gaussian Kernel
kernel = 2 * np.math.ceil(2 * sigma) + 1  # Kernel size

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('blur', gray_image)
cv2.waitKey(0)
blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
gray_image = cv2.subtract(gray_image, blur)
cv2.imshow('blur', blur)
cv2.waitKey(0)
cv2.imshow('gray_image', gray_image)

# compute sobel response
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
mag = np.hypot(sobelx, sobely)
ang = np.arctan2(sobely, sobelx)

# threshold
threshold = 4 * fudgefactor * np.mean(mag)
mag[mag < threshold] = 0

# either get edges directly
if with_nmsup is False:
    mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    gray = cv2.bitwise_not(result)
    # img = gray.img_to_array(gray, dtype='uint8')
    #
    # bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    cv2.imshow('im', result)
    cv2.waitKey()

# or apply a non-maximal suppression
else:

    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('shapes', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if len(approx) > 5 and area > 1000:
            cv2.drawContours(image, [approx], 0, (0, 0, 0), 5)

            x = approx.ravel()[0]
            y = approx.ravel()[1] - 5
            if len(approx) == 3:
                cv2.putText(image, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspectRatio = float(w) / h
                print(aspectRatio)
                if 0.95 <= aspectRatio < 1.05:
                    cv2.putText(image, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                else:
                    cv2.putText(image, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

            elif len(approx) == 5:
                cv2.putText(image, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            elif len(approx) == 10:
                cv2.putText(image, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            else:
                cv2.putText(image, "hole", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('shapes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # non-maximal suppression
    mag = orientated_non_max_suppression(mag, ang)
    # create mask
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('im', result)
    cv2.waitKey()
