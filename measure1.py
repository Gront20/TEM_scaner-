
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2
import time
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
i2 = []
pix = []
for i1 in range(0, 255):
    ret, thre = cv2.threshold(image, i1, 255,  cv2.THRESH_BINARY)
    n_white_pix = np.sum(thre == 0)
    i2.append(i1)
    pix.append(n_white_pix)
ret, thre = cv2.threshold(image, 145, 255,  cv2.THRESH_BINARY)
cv2.imshow ( "Image", thre )
df = [0 for i in range(0,255)]
for i in range(1, 254):
    df[i] = (int((pix[i + 1] - pix[i - 1]) / (2 * 1)))
plt.plot(i2, pix)
#plt.plot(i2, df)
plt.show()
gray = cv2.cvtColor(thre, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(thre, 3)
cv2.imshow ( "Image", gray )
edged = cv2.Canny(gray, 1, 1)

#edged = cv2.dilate(edged, None, iterations=1)
#edged = cv2.erode(edged, None, iterations=1)
cv2.waitKey ( 0 )
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None
i = 0
lstB = []
lstA = []
area = 0
Area1 = []
BoxArray = []

#pixel = (50, 242, 65)
#gray[25,25] = pixel
#for x in range(50,125):
#    for y in range(50,125):
#        gray[x, y] = pixel
#cv2.imshow("Image", gray)
#cv2.waitKey(0)


# def scaner(x, y, n):
#     flag = False
#     for index in range(x, x + n):
#         if gray[index][y][0] > 0 and gray[index][y][1] > 0 and gray[index][y][2] > 0:
#             return False
#         if gray[index][y + n][0] > 0 and gray[index][y + n][1] > 0 and gray[index][y + n][2] > 0:
#             return False
#     for index in range(y, y + n):
#         if gray[x][index][0] > 0 and gray[x][index][1] > 0 and gray[x][index][2] > 0:
#             return False
#         if gray[x + n][index][0] > 0 and gray[x + n][index][1] > 0 and gray[x + n][index][2] > 0:
#             return False
#     return True
#
# pixel = (0, 0, 0)
#
# def ClearScan(x, y, n):
#     for index1 in range(x, x + n):
#         for index2 in range(y, y + n):
#             gray[index1][index2] = pixel
#
# for x in range(0, 200):
#     print(x)
#     for y in range(500, 700):
#         if scaner(x, y, 6) == True:
#             ClearScan(x, y, 6)
#
#
# cv2.imshow( "Image", gray )
# cv2.waitKey ( 0 )


for c in cnts:
    orig = gray.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)
    area = cv2.contourArea(c)
    BoxArray.append(box)
    Area1.append(area)
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 3, (0, 0, 255), -1)
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint ( tl, tr )
    (blbrX, blbrY) = midpoint ( bl, br )
    (tlblX, tlblY) = midpoint ( tl, bl )
    (trbrX, trbrY) = midpoint ( tr, br )
    cv2.circle ( orig, (int ( tltrX ), int ( tltrY )), 1, (255, 0, 0), -1 )
    cv2.circle ( orig, (int ( blbrX ), int ( blbrY )), 1, (255, 0, 0), -1 )
    cv2.circle ( orig, (int ( tlblX ), int ( tlblY )), 1, (255, 0, 0), -1 )
    cv2.circle ( orig, (int ( trbrX ), int ( trbrY )), 1, (255, 0, 0), -1 )
    cv2.line ( orig, (int ( tltrX ), int ( tltrY )), (int ( blbrX ), int ( blbrY )), (255, 0, 255), 1 )
    cv2.line ( orig, (int ( tlblX ), int ( tlblY )), (int ( trbrX ), int ( trbrY )), (255, 0, 255), 1 )
    dA = dist.euclidean ( (tltrX, tltrY), (blbrX, blbrY) )
    dB = dist.euclidean ( (tlblX, tlblY), (trbrX, trbrY) )
    #print ( dA )
    #print(dB)
    cv2.imshow( "Image", orig )

print(len(Area1))
minAP = 100000
maxAP = 0
for index1 in range(0, len(Area1)):
    if minAP > Area1[index1]:
        minAP = Area1[index1]
    if maxAP < Area1[index1]:
        maxAP = Area1[index1]
delta = (maxAP-minAP)/100
print(delta)
print(minAP)
print(maxAP)
minAP = 0
ArrayInter = []
Diap = [0 for i in range(0, 100)]
DiapS = [i*delta for i in range(0, 100)]
for index1 in range(0, 100):
    for j in range(41, len(Area1)):
        if Area1[j] >= minAP + delta*index1 and Area1[j] < minAP + delta * index1 + delta:
            Diap[index1] = Diap[index1] + 1
            if Diap[index1] > 2000:
                 Diap[index1] = 20600
            if Area1[j] < 2:
                print(BoxArray[j])
            if index1 < 3:
                BA = np.array ( BoxArray[j] ).reshape ( (-1, 1, 2) ).astype ( np.int32 )
                cv2.fillPoly(orig, pts = [BA], color=(0, 0, 0))

def scaner(x, y, n):
    flag = False
    for index in range(x, x + n):
        if gray[index][y][0] > 0 and gray[index][y][1] > 0 and gray[index][y][2] > 0:
            return False
        if gray[index][y + n][0] > 0 and gray[index][y + n][1] > 0 and gray[index][y + n][2] > 0:
            return False
    for index in range(y, y + n):
        if gray[x][index][0] > 0 and gray[x][index][1] > 0 and gray[x][index][2] > 0:
            return False
        if gray[x + n][index][0] > 0 and gray[x + n][index][1] > 0 and gray[x + n][index][2] > 0:
            return False
    return True

pixel = (0, 0, 0)

def ClearScan(x, y, n):
    for index1 in range(x, x + n):
        for index2 in range(y, y + n):
            gray[index1][index2] = pixel

for x in range(0, 900):
    print(x)
    for y in range(0, 900):
        if scaner(x, y, 4) == True:
            ClearScan(x, y, 4)


plt.plot(DiapS, Diap)
plt.show()
cv2.imshow( "Image", gray )
cv2.waitKey ( 0 )

edged = cv2.Canny(gray, 1, 1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    area = cv2.contourArea ( c )
    Area1.append(area)
    file = open ( "dimPor.txt", "w" )
for index in range ( len ( Area1 ) ):
    if Area1[index] >= 11:
        file.write ( str ( "{:.4f}".format ( Area1[index] ) ) + "\n" )
file.close ( )
