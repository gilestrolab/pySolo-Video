__author__ = 'quentin'

from pysolovideo.tracking.cameras import MovieVirtualCamera
from pysolovideo.tracking.trackers import *
import traceback
# Build ROIs from greyscale image
from pysolovideo.tracking.roi_builders import IterativeYMaze, DefaultROIBuilder

# the robust self learning tracker
from pysolovideo.tracking.trackers import AdaptiveBGModelOneObject

# the standard monitor
from pysolovideo.tracking.monitor import Monitor
from pysolovideo.utils.io import SQLiteResultWriter

from pysolovideo.tracking.roi_builders import TargetGridROIBuilderBase
import pkg_resources
import optparse
import logging
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
    def _find_position(self, img, mask,t):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._track(img, grey, mask, t)


    def _track(self, img,  grey, mask, t):
        if self._accum is None:
            self._accum = grey.astype(np.float64)

        frame_float64 = grey.astype(np.float64)
        cv2.accumulateWeighted(frame_float64, self._accum, self._alpha)
        bg = self._accum.astype(np.uint8)

        diff = cv2.subtract(bg, grey)
        cv2.medianBlur(grey, 7, grey)
        _, bin_im = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(bin_im,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

        contours= self._filter_contours(contours)

        if len(contours) != 1:
            raise NoPositionError
        hull = contours[0]

        (_,_) ,(w,h), angle = cv2.minAreaRect(hull)

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


# _result_dir = "/psv_data/results/"
# _last_img_file = "/tmp/psv/last_img.jpg"
# _dbg_img_file = "/tmp/psv/dbg_img.png"
# _log_file = "/tmp/psv/psv.log"

if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option("-o", "--output", dest="out", help="the output file (eg out.csv   )", type="str",default=None)
    parser.add_option("-i", "--input", dest="input", help="the output video file", type="str")
    #
    parser.add_option("-r", "--result-video", dest="result_video", help="the path to an optional annotated video file."
                                                                "This is useful to show the result on a video.",
                                                                type="str", default=None)

    parser.add_option("-d", "--draw-every",dest="draw_every", help="how_often to draw frames", default=0, type="int")

    parser.add_option("-m", "--mask", dest="mask", help="the mask file with 3 targets", type="str")

    (options, args) = parser.parse_args()

    option_dict = vars(options)

    logging.basicConfig(level=logging.INFO)


    logging.info("Starting Monitor thread")

    cam = MovieVirtualCamera(option_dict ["input"], use_wall_clock=False)

    my_image = cv2.imread(option_dict['mask'])
    print option_dict['mask']
    # cv2.imshow('window', my_image)
    roi_builder = Ymaze()
    # roi_builder = IterativeYMaze()
    rois = roi_builder(my_image)

    logging.info("Initialising monitor")

    cam.restart()

    metadata = {
                             "machine_id": "None",
                             "machine_name": "None",
                             "date_time": cam.start_time, #the camera start time is the reference 0
                             "frame_width":cam.width,
                             "frame_height":cam.height,
                             "version": "whatever"
                              }
    draw_frames = False
    if option_dict["draw_every"] > 0:
        draw_frames = True


    monit = Monitor(cam, YMazeTracker, rois,
                    draw_every_n=option_dict["draw_every"],
                    draw_results=draw_frames,
                    video_out=option_dict["result_video"],
                    )

    try:
        if option_dict["out"] is None:
            monit.run(None)
        else:
            with SQLiteResultWriter(option_dict["out"],rois, metadata) as rw:
                logging.info("Running monitor" )
                monit.run(rw)
    except KeyboardInterrupt:
        logging.info("Keyboard Exception, stopping monitor")
        monit.stop()
    except Exception as e:
        print traceback.format_exc(e)


    logging.info("Stopping Monitor")
