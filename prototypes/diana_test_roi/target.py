__author__ = 'diana'

from pysolovideo.tracking.roi_builders import  TargetGridROIBuilderBase
import cv2
class Ymaze(TargetGridROIBuilderBase):
    _vertical_spacing =  0
    _horizontal_spacing =  0
    _n_rows = 1
    _n_cols = 1
    _horizontal_margin_left = .0 # from the center of the target to the external border (positive value makes grid larger)
    _horizontal_margin_right = .0
    _vertical_margin_top = 0 # from the center of the target to the external border (positive value makes grid larger)
    _vertical_margin_bottom = 0

myYmaze = Ymaze()
myImage = cv2.imread("/home/diana/Desktop/experiment_4_mazes_tuesday_11_08_2015/monitor003_test.png")
rois = myYmaze(myImage)

r = rois[0]

cv2.drawContours(myImage, [r.polygon],-1, (0,0,0), 3, cv2.CV_AA)
cv2.drawContours(myImage, [r.polygon],-1, (255,255,0), 1, cv2.CV_AA)

cv2.imshow("my_roi",myImage)
cv2.waitKey(-1)