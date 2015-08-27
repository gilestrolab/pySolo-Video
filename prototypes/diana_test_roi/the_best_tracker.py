__author__ = 'diana'

import cv2
import numpy as np
from pysolovideo.tracking.cameras import MovieVirtualCamera
from pysolovideo.tracking.trackers import *
MY_VIDEO = "/home/diana/Desktop/males_Tuesday_11_08_2015/monitor_3/2015-08-11-Aug-13-1439288019-GGSM-003-DAM15-FLY09.mp4"

cap = MovieVirtualCamera(MY_VIDEO)

accum = None
# i = 0
# for frame_idx, (t, f) in enumerate(cap):
#     if frame_idx % 100 != 0:
#         continue
#     if frame_idx >  2000:
#         break
#     print frame_idx, i
#     grey = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
#
#     if accum is None:
#         accum = np.zeros_like(grey, dtype=np.float64)
#
#     frame_float64 = grey.astype(np.float64)
#
#     i += 1
#
#
#     cv2.accumulate(frame_float64, accum)
#     cv2.imshow("grey", grey)
#     cv2.imshow("my_img", accum/float(i)/256.0)
#     cv2.waitKey(1)
#
#
# bg = accum/float(i)
# bg = bg.astype(np.uint8)

# cap = MovieVirtualCamera(MY_VIDEO)
# ALPHA = 0.001


#
# for frame_idx, (t, f) in enumerate(cap):
#
#     grey = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
#
#     if accum is None:
#         accum = grey.astype(np.float64)
#
#     frame_float64 = grey.astype(np.float64)
#     cv2.accumulateWeighted(frame_float64, accum, ALPHA)
#
#     bg = accum.astype(np.uint8)
#     cv2.imshow("grey", grey)
#     cv2.imshow("bg", bg)
#
#
#
#     diff = cv2.subtract(bg, grey)
#     cv2.medianBlur(grey, 7, grey)
#     _, bin_im = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
#
#     cv2.imshow("bin", bin_im)
#
#     contours,hierarchy = cv2.findContours(bin_im,
#                                           cv2.RETR_EXTERNAL,
#                                           cv2.CHAIN_APPROX_SIMPLE)
#     contours = filter_contours(contours)
#     # if len(contours) > 1:
#     #     raise Exception("TODO") # TODO
#     cv2.drawContours(f, contours, -1, (0,0,255), 1, cv2.CV_AA)
#     #cv2.imshow("grey", grey)
#     cv2.imshow("out", f)
#     #cv2.imshow("diff", diff)
#
#
#     cv2.waitKey(1)
#



class YMazeTracker(BaseTracker):

    def __init__(self, roi, data=None):
        self._accum = None
        self._alpha = 0.001
        super(YMazeTracker, self).__init__(roi, data)

    def _filter_contours(self, contours, min_area =50, max_area=200):
        out = []
        for c in contours:
            if c.shape[0] < 6:
                continue
            area = cv2.contourArea(c)
            if not min_area < area < max_area:
                continue

            out.append(c)
        return out

    def _track(self, img,  grey, mask,t):

        if self._accum is None:
            self._accum = grey.astype(np.float64)

        frame_float64 = grey.astype(np.float64)
        cv2.accumulateWeighted(frame_float64, accum, self._alpha)

        bg = accum.astype(np.uint8)
        # cv2.imshow("grey", grey)
        # cv2.imshow("bg", bg)

        diff = cv2.subtract(bg, grey)
        cv2.medianBlur(grey, 7, grey)
        _, bin_im = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # cv2.imshow("bin", bin_im)

        contours,hierarchy = cv2.findContours(bin_im,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        hull = self._filter_contours(contours)

        (_,_) ,(w,h), angle  = cv2.minAreaRect(hull)

        M = cv2.moments(hull)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        if w < h:
            angle -= 90
            w,h = h,w

        angle = angle % 180

        h_im = min(grey.shape)
        max_h = 2*h_im
        if w>max_h or h>max_h:
            raise NoPositionError

        x_var = XPosVariable(int(round(x)))
        y_var = YPosVariable(int(round(y)))
        w_var = WidthVariable(int(round(w)))
        h_var = HeightVariable(int(round(h)))
        phi_var = PhiVariable(int(round(angle)))

        out = DataPoint([x_var, y_var, w_var, h_var, phi_var])

        return out

