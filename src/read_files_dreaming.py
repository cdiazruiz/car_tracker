"""
util/detections.py

Utilities for reading the detection, calibration, and odometry files provided with the KITTI dataset.

Carlos Diaz-Ruiz cad297@cornell.edu
Cornell University Autonomous Systems Lab
"""

import csv
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# from typing import NamedTuple
# from collections import namedtuple


class Detection(object):
    """
    Class for representing bounding box detections, using the same
    parameters provided in KITTI specifications.

    """
    """
    frame: int
    track_id: int
    type: str ('Car', 'Van', 'Truck', 'Pedestrian', 'Person Sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'
    truncated: int (0,1,2 depending on level of truncation)
    occluded: int (0,1,2,3) 0= fully visible, 1 = partly occluded, 2= largely occluded, 3 = unknown
    alpha: float observation angle of object [-pi..pi]
    bb_left: float
    bb_top: float
    bb_right: float
    bb_bottom: float
    height: float (meters)
    width: float (meters)
    length: float (meters)
    x: float (meters in camera coordinates forward)
    y: float (meters in camera coordinates)
    z: float (meters in camera coordinates)
    rot_y: float [rotation around y-axis in camera coordinates [-pi..pi]
    score: float only for results: float, indicating confidence in detection, needed for p/r curves, higher is better
    number: detection number (may be unnused)
    """

    def __init__(self, frame, track_id, type, truncated, occluded, alpha, bb_left, bb_top, bb_right, bb_bottom,
                 height, width, length, x, y, z, rot_y, score, num):
        self.frame = int(frame)
        self.track_id = int(track_id)
        self.type = str(type)
        self.bb_left = float(bb_left)
        self.bb_top = float(bb_top)
        self.bb_right = float(bb_right)
        self.bb_bottom = float(bb_bottom)
        self.height = float(height)
        self.width = float(width)
        self.length = float(length)
        self.conf = float(width)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.alpha = float(alpha)
        self.rot_y = float(rot_y)
        self.score = float(score)
        self.number = int(num)  # detection number (may be unused)
        self.color = np.random.rand(1,3) #for plotting purposes
    @property
    def bb_width(self):
        """

        :return: float
        width of image bounding box in pixels

        """
        return np.abs(self.bb_right -self.bb_left)

    @property
    def bb_height(self):
        """

        :return: float
        width of image bounding box in pixels

        """
        return np.abs(self.bb_bottom - self.bb_top)


class Odometry(object):
    """
    Class for representing odometry readings using the same
    parameters provided in KITTI specifications.

    GPS/IMU 3D localization unit
    ============================

    The GPS/IMU information is given in a single small text file which is
    written for each synchronized frame. Each text file contains 30 values
    which are:

      - lat:     latitude of the oxts-unit (deg)
      - lon:     longitude of the oxts-unit (deg)
      - alt:     altitude of the oxts-unit (m)
      - roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
      - pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
      - yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
      - vn:      velocity towards north (m/s)
      - ve:      velocity towards east (m/s)
      - vf:      forward velocity, i.e. parallel to earth-surface (m/s)
      - vl:      leftward velocity, i.e. parallel to earth-surface (m/s)
      - vu:      upward velocity, i.e. perpendicular to earth-surface (m/s)
      - ax:      acceleration in x, i.e. in direction of vehicle front (m/s^2)
      - ay:      acceleration in y, i.e. in direction of vehicle left (m/s^2)
      - az:      acceleration in z, i.e. in direction of vehicle top (m/s^2)
      - af:      forward acceleration (m/s^2)
      - al:      leftward acceleration (m/s^2)
      - au:      upward acceleration (m/s^2)
      - wx:      angular rate around x (rad/s)
      - wy:      angular rate around y (rad/s)
      - wz:      angular rate around z (rad/s)
      - wf:      angular rate around forward axis (rad/s)
      - wl:      angular rate around leftward axis (rad/s)
      - wu:      angular rate around upward axis (rad/s)
      - posacc:  velocity accuracy (north/east in m)
      - velacc:  velocity accuracy (north/east in m/s)
      - navstat: navigation status
      - numsats: number of satellites tracked by primary GPS receiver
      - posmode: position mode of primary GPS receiver
      - velmode: velocity mode of primary GPS receiver
      - orimode: orientation mode of primary GPS receiver

    """
    def __init__(self, vf, vl, vu, wf, wl, wu, mode,frame):
        self.vf = float(vf)
        self.vl = float(vl)
        self.vu = float(vu)
        self.wf = float(wf)
        self.wl = float(wl)
        self.wu = float(wu)
        self.mode = int(mode)
        self.frame = int(frame)
    # @property
    # def bb_x(self):
    #     """
    #     Returns
    #     -------
    #     float
    #         x-coordinate of the center of the bounding box, in the image frame
    #         The image frame x-axis increases in a left-to-right direction.
    #     """
    #     return (self.bb_left + self.bb_width / 2)
    #
    # @property
    # def bb_y(self):
    #     """
    #
    #     Returns
    #     -------
    #     float
    #         y_coordinate of the center of the bounding box, in the image frame
    #         The image frame y-axis increases in a top-to-bottom direction
    #     """
    #     return (self.bb_top + self.bb_height / 2)
    #
    # @property
    # def top_left(self):
    #     """
    #     Returns the point corresponding to the top-left corner of the
    #     detection bounding box.
    #
    #     Returns
    #     -------
    #     (int, int)
    #
    #     """
    #     return (self.bb_left, self.bb_top)
    #
    # @property
    # def bottom_right(self):
    #     """
    #     Returns the point corresponding to the bottom-right corner of hte
    #     detection bounding box.
    #
    #     Returns
    #     -------
    #     (int, int)
    #
    #     """
    #     x = int(np.rint(self.bb_left + self.bb_width))
    #     y = int(np.rint(self.bb_top + self.bb_height))
    #     return (x, y)
    #
    # def draw(self, image, thickness=2, color=(0, 255, 0), label=""):
    #     cv2.rectangle(image, self.top_left, self.bottom_right, thickness=thickness,
    #                   color=color)
    #     if label:
    #         label = str(label)
    #         pos = (int(np.rint(self.bb_x - 10)), int(np.rint(self.bb_y)))
    #         cv2.putText(image, text=label, org=pos,
    #                     thickness=2, color=color,
    #                     fontFace=cv2.FONT_HERSHEY_DUPLEX,
    #                     fontScale=0.8)
    #
    # def show_conf(self, image, color=(0, 255, 0)):
    #     cv2.putText(image, text="conf=%.2f" % self.conf,
    #                 org=(self.bb_left, self.bb_top - 5), color=color,
    #                 fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75)


# class ReidDetection(Detection):
#     """
#     Detection that has been augmented by also including a re-ID feature vector.
#     Fields are same as the Detection class, with the addition of:
#
#     feature : numpy ndarray
#         1D feature vector
#
#     """
#
#     def __init__(self, detection, feature):
#         super().__init__(frame=detection.frame, person_id=detection.person_id,
#                          bb_left=detection.bb_left, bb_top=detection.bb_top,
#                          bb_width=detection.bb_width, bb_height=detection.bb_height,
#                          conf=detection.conf, num=detection.number, ori=detection.ori)
#         self.feature = feature


def read_detections_from_kitti(filename, num_frames=None, gt=False, challenge=0):
    """
    Read a text file containing data on detections, and process it into a
    list of Detection objects.

    Parameters
    ----------
    filename : str
        Name of the detections file. Should list a series of detections in
        the format specified by https://motchallenge.net/instructions/

    num_frames : int
        Number of frames in the video corresponding to the detection file

    gt : bool
        Whether the detections file is ground truth
    challenge : bool
        true - from the tracking challenge format (includes frame and id number in first two entries)
        false - from the object challenge format
    Returns
    -------
    dict
        Dictionary whose keys are integers and values are lists. Keys are
        frame id_number numbers, and values are lists of Detection objects that
        represent the detections which occurred at the corresponding frame.
        Some of the lists may be empty, if no detections occurred at certain
        frames in the image sequence.
    """

    # raise ValueError('Line 167 detections.py')
    if challenge:
        reader = csv.reader(open(filename, "r", newline=""), delimiter=' ')
        converted_lines = []
        det_num = 0
        for line_num, line in enumerate(reader):
            converted_line = []
            for i in range(len(line)):
                if i < 2:
                    converted_line.append(int(line[i]))
                elif i == 2:
                    converted_line.append(str(line[i]))
                elif 2 < i < 4:
                    converted_line.append(int(line[i]))
                else:
                    converted_line.append(float(line[i]))
            if gt:
                converted_line.append(int(0))  #ground truth txt files do not include score so append it
            converted_line.append(det_num)
            converted_lines.append(converted_line)
            det_num += 1
    else:
        filenames = [f for f in listdir(filename) if isfile(join(filename, f))]
        det_num = 0
        converted_lines=[]
        print('path', filename)
        for file in filenames:
            reader = csv.reader(open(os.path.join(filename,file),"r", newline=""), delimiter=' ')
            frame_num=int(file[-10:-4])
            for line_num, line in enumerate(reader):
                converted_line = [frame_num, 0]
                for i in range(len(line)):
                    if i == 0:
                        converted_line.append(str(line[i]))
                    elif 1 <= i <= 2:
                        converted_line.append(int(line[i]))
                    else:
                        converted_line.append(float(line[i]))
                converted_line.append(det_num)
                converted_lines.append(converted_line)
                det_num += 1
    detections = [Detection(*line) for line in converted_lines]
    if num_frames is None:
        # If no frame count was provided, use the frame number of the last
        # detection as the frame count.
        num_frames = detections[-1].frame+1
    elif num_frames < 0 or type(num_frames) is not int:
        raise ValueError("Number of frames must be a non-negative integer")
    detection_time_series = [[] for i in range(num_frames)]
    for det in detections:
        k = det.frame  # frames are 1-indexed, so subtract 1 for list index
        detection_time_series[k].append(det)
    return detection_time_series


def read_odometry_from_kitti(filename):
    """
    Read a text file containing odometry, and process it into a
    list of odometry objects.

    Parameters
    ----------
    filename : str
        Name of the odometry file. Should list a series of detections in
        the format specified by https://motchallenge.net/instructions/

    Returns
    -------
    dict
        Dictionary whose keys are integers and values are lists. Keys are
        frame id_number numbers, and values are lists of Detection objects that
        represent the detections which occurred at the corresponding frame.
        Some of the lists may be empty, if no detections occurred at certain
        frames in the image sequence.
    """

    reader = csv.reader(open(filename, "r", newline=""), delimiter=' ')
    converted_lines = []
    for line_num, line in enumerate(reader):
        converted_line = []
        line.pop()
        for i in range(len(line)):
            if 7 < i < 11:
                converted_line.append(float(line[i]))
            elif 19 < i < 23:
                converted_line.append(float(line[i]))
            elif i == len(line)-1:
                converted_line.append(int(float(line[i])))
        converted_line.append(line_num)
        converted_lines.append(converted_line)
    odometry_list = [Odometry(*line) for line in converted_lines]

    return odometry_list


def read_calibration_from_kitti(filename):
    """
    Read a text file containing calibration parameters, and process it into a
    list of odometry objects.

    Parameters
    ----------
    filename : str
        Name of the odometry file. Should list a series of detections in
        the format specified by https://motchallenge.net/instructions/

    Returns
    -------
    dict
        Dictionary whose keys are  (P0-P3, Rect, Tvelo2ca, Timu2vel ) and values matrices.
    """

    reader = csv.reader(open(filename, "r", newline=""), delimiter=' ')
    transform_mats={}
    for line_num, line in enumerate(reader):
        converted_line = []
        # line.pop()  #the last two values in line are a space so I am removing them
        # line.pop()
        for i in range(len(line)):
            if i == 0:
                converted_line.append(str(line[i][0:-1]))
            else:
                converted_line.append(float(line[i]))
        # print(converted_line)
        if len(converted_line[1:]) == 9:
            transform_mats[converted_line[0]] = np.reshape(np.array(converted_line[1:]), (-1, 3))
        elif len(converted_line[1:]) == 12:
            row1=np.array([0, 0, 0, 1])
            transform_mats[converted_line[0]] = np.vstack((np.reshape(np.array(converted_line[1:]), (-1, 4)), row1))

    return transform_mats
