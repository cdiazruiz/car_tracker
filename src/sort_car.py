"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function
import math
from matplotlib import gridspec
from matplotlib import path
from operator import add
# from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.stats.distributions import chi2
from scipy.spatial import distance
import glob
import matplotlib.image as mpimg
import time
import argparse
# from filterpy.kalman import KalmanFilter
from read_files_dreaming import read_detections_from_kitti, read_odometry_from_kitti, read_calibration_from_kitti
from transformations import convert_vel
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



def wrapangle(angle):
    """

    :param angle:
    :return: an angle between -pi and pi
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def error_ellipse(ax, xc, yc, cov, sigma=1, **kwargs):
    '''
    Plot an error ellipse contour over your data.
    Inputs:
    ax : matplotlib Axes() object
    xc : x-coordinate of ellipse center
    yc : x-coordinate of ellipse center
    cov : covariance matrix
    sigma : # sigma to plot (default 1)
    additional kwargs passed to matplotlib.patches.Ellipse()
    '''
    w, v = np.linalg.eigh(cov) # assumes symmetric matrix
    order = w.argsort()[::-1]
    w, v = w[order], v[:,order]
    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    ellipse = Ellipse(xy=(xc,yc),
                    width=2.*sigma*np.sqrt(w[0]),
                    height=2.*sigma*np.sqrt(w[1]),
                    angle=theta, **kwargs)
    ellipse.set_facecolor('none')
    ax.add_artist(ellipse)

def mahab_dist(trk, det):
    '''

        Inputs:
        trk : track object
        det : detection
        Outputs:
        dist : mahalanobis distance based only on position
        Description: mahalanobis distance is caluculated in measurement space, only in x and y positions
        '''

    # print('det in mahab', det)
    # det_vec = [det[0], det[1], det[2], det[3], det[4]]
    [x, y, th, s, l, w]=trk.state
    c = 0.25
    z_pred = np.array([x - (l*c) * np.cos(th), y - (l*c) * np.cos(th), th, l, w])
    H = np.array([[1, 0, l * c * np.sin(th), 0, c * np.cos(th), 0],
                  [0, 1, l * c * np.cos(th), 0, -c * np.cos(th), 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    S = H.dot(trk.P.dot(H.T))+trk.R1
    v=np.array(det[0:5]).flatten()-z_pred
    v[2]=wrapangle(v[2])
    dist = v[0:2].dot(np.linalg.inv(S[0:2,0:2]).dot(v[0:2].T))
    return dist


class KalmanCarTracker(object):
    """
    This class represents the internel state of individual tracked cars.
    The state is z (forward pos), x (right pos), th (rel orientation), s (abs groundspeed), l (length), w (width)
    """
    count = 0

    def __init__(self, init_det, dT=0.1, sigma_R=1, tracker_id=0,
                 active=True, initial_time=None,sigma_Q = 1):
        """
        Initialises a tracker using initial bounding box.
        """
        self.t = 0.1
        self.initialize = False
        s=0  #initial speed is set to zero but will be updated with numerical differentiation
        self.c = 0.25  # 1/4 for estimate of length between center and back axle
        x_c=init_det[0]
        y_c=init_det[1]
        th=init_det[2]
        l=init_det[3]
        w=init_det[4]

        x = x_c - l * self.c * np.cos(th)
        y = y_c - l * self.c * np.sin(th)
        self.state = np.array([x, y, th, s, l, w])
        self.dT = dT


        # create lists for storing histories (detections, odometry and tracks)
        # length of histories
        self.det_hist = []
        self.det_hist.append(init_det)
        self.history = []
        self.odo_hist = []
        self.P = np.identity(len(self.state))

        #error covariance matrix initialization
        self.P[0,0] = 6
        self.P[1,1] = 6
        self.P[2,2] = 0.1
        self.P[3,3] = 5**2
        self.P[4,4] = 0.5
        self.P[5,5] = 0.3

        #measuremnet noise covariance matrix (diagonal)
        self.R1 = np.identity(5)
        self.R1[0,0]=0.3
        self.R1[1,1]=0.3
        self.R1[2,2]=0.03
        self.R1[3,3]=0.07
        self.R1[4,4]=0.04

        self.Q = np.identity(7)  # th s x_vel y_vel wz el ew
        self.c = 0.25 #1/4 for estimate of length between center and back axle

        #process noise covariance matrix
        self.Q[0,0] = 0.030461174*10
        # self.Q[1,1] = 1
        # self.Q[2,2] = 0.0054783
        # self.Q[3,3] = 0.00544783
        # self.Q[4,4] = 0.030461
        self.Q[5,5] = 0.01
        self.Q[6,6] = 0.01
        self.Phist=[]
        self.color=np.random.rand(1,3) #used only for display
        self.time_since_update = 0
        self.id = KalmanCarTracker.count
        KalmanCarTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    def s_estimate(self):

        #estimates ground speed by using two associated detections in a row.
        #used when initializing a track.
        last_det=self.det_hist[-1]
        pen_det=self.det_hist[-2]
        # frame_diff=last_det.frame-pen_det.frame

        dt=(self.time_since_update+1)*self.dT
        print('dt for s estimate', dt)
        odo=self.odo_hist[-2]
        vx=odo[0]
        vy=odo[1]
        wz=odo[2]
        x_term=(last_det[0]-pen_det[0])/dt + vx - pen_det[1]*wz
        y_term=(last_det[1]-pen_det[1])/dt + vy + pen_det[0]*wz
        s=np.sqrt(x_term**2+y_term**2)
        th= math.atan2(x_term,y_term)
        if s<2.2:
            s=0
            th = last_det[2]
        return s, th


    def predict(self, odometry, calib):
        """
        :param odometry:
        :param calib:
        :return: kalman prediction
        """
        #kalman predict
        if self.age==0:
            self.odo_hist.append(odometry)
        [x, y, th, s, l, w] = self.state
        vx = odometry[0]
        vy = odometry[1]
        wz = odometry[2]
        # print('odometry', odometry)
        # print('state', self.state)
        t=self.t
        x += t*(s*np.cos(th)-vx+y*wz)
        y += t*(s*np.sin(th)-vy-x*wz)
        th += -t*wz
        th = wrapangle(th)
        Ak = np.array([[1, wz * t, -t * s * np.sin(th), -t * np.cos(th), 0, 0],
                       [-wz * t, 1, t * s * np.cos(th), t * np.sin(th), 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
        Wk = np.array([[-(s * t ** 2 * np.sin(th)) / 2, (t ** 2 * np.cos(th)) / 2, -t, -(t ** 2 * wz) / 2,
                         t * y - (t ** 2 * wz * x) / 2, 0, 0],
                       [(s * t ** 2 * np.cos(th)) / 2, (t ** 2 * np.sin(th)) / 2, (t ** 2 * wz) / 2, -t,
                        - t * x - (t ** 2 * wz * y) / 2, 0, 0],
                       [t, 0, 0, 0, 0, 0, 0],
                       [0, t, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, t, 0],
                       [0, 0, 0, 0, 0, 0, t]])

        self.P = Ak.dot(self.P).dot(Ak.T) + Wk.dot(self.Q).dot(Wk.T)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak= 0
        self.time_since_update += 1
        self.state = np.array([x, y, th, s, l, w])
        # print('predicted state', self.state)

    def calc_corners(self):

        [x, y, th, s, l, w] = self.state
        cth=np.cos(th)
        sth=np.sin(th)
        c1 = np.array([x + (3 / 4) * l * cth - (1 / 2) * w * sth, y + (3 / 4) * l * sth + (1 / 2) * w * cth])
        c2 = np.array([x - (1 / 4) * l * cth - (1 / 2) * w * sth, y - (1 / 4) * l * sth + (1 / 2) * w * cth])
        c3 = np.array([x - (1 / 4) * l * cth + (1 / 2) * w * sth, y - (1 / 4) * l * sth - (1 / 2) * w * cth])
        c4 = np.array([x + (3 / 4) * l * cth + (1 / 2) * w * sth, y + (3 / 4) * l * sth - (1 / 2) * w * cth])
        #% center points
        m1 =np.array([x + (3 / 4) * l * cth, y + (3 / 4) * l * sth])
        m2 = np.array([x + (1 / 4) * l * cth - (1 / 2) * w * sth, y + (1 / 4) * l * sth + (1 / 2) * w * cth])
        m3 = np.array([x - (1 / 4) * l * cth + (1 / 2), y - (1 / 4) * l * sth])
        m4 = np.array([x + (1 / 4) * l * cth + (1 / 2) * w * sth, y + (1 / 4) * l * sth - (1 / 2) * w * cth])
        pos =np.vstack((c1,c2,c3,c4,m1,m2,m3,m4))
        return pos

    def which_rmin(self):
        '''
        determines which corner would yield minimum range based on state
        :return:
        '''
        #TODO: implement code
        positions = self.calc_corners()
        idx_rmin = np.argmin(np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2))
        rmin = min(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        idx_rmax = np.argmax(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        rmax = max(np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2))
        return [rmin, idx_rmin, rmax, idx_rmax]

    def occluding_angles(self):
        '''
        determine which corners would determine occluding angles
        :return:
        '''
        pos = self.corners[0:4,0:2]
        not_visible = self.visibility()
        bearings = np.arctan2(pos[1,:],pos[0,:])
        bearings[~not_visible] =-1000
        bmax=max(bearings)
        bmax_id=np.argmax(bearings)
        bearings[~not_visible] = 1000
        bmin=min(bearings)
        bmin_id=np.argmin(bearings)
        return np.array([bmin, bmin_id, bmax, bmax_id])


    def visibility(self):
        '''
        determines which corners are visible
        :return:
        '''
        pos = self.corners[0:4,0:2]
        pointlist=[]
        scale=0.999
        # for i in range(len(pos)):
            # pointlist.append(Point(pos[i,0]*scale,pos[i,1]*scale))
        rectangle = path.Path([(pos[0, 0], pos[0, 1]), (pos[1, 0], pos[1, 1]), (pos[2, 0], pos[2, 1]), (pos[3, 0], pos[3, 1])])
        not_visible= rectangle.contains_points(pos[0:4,:]*scale)
        # polygon = Polygon([(pos[0,0],pos[0,1]),(pos[1,0],pos[1,1]),(pos[2,0],pos[2,1]),(pos[3,0],pos[3,1])])
        # print(polygon.contains(point))
        return not_visible


    def update(self, dets, odometry):
        """
        Updates the state vector with observed 3D bbox.
        """
        [x, y, th, s, l, w] = self.state
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.det_hist.append(dets)
        self.odo_hist.append(odometry)
        det_vec = [dets[0], dets[1], dets[2], dets[3], dets[4]]
        if len(self.det_hist)>15:
            self.det_hist.pop(0)
            self.odo_hist.pop(0)
        if self.age<=4 and  1<=self.hits<=2: #happens for the f
            #initial speed is set to zero but will be updated with numerical differentiaon
            self.c = 0.25  # 1/4 for estimate of length between center and back axle
            x_c=dets[0]
            y_c=dets[1]
            l_car=dets[3]
            # th = dets.rot_y
            x = x_c - l_car * self.c * np.cos(th)
            y = y_c - l_car * self.c * np.sin(th)
            s, th = self.s_estimate()
            print('s estimate', s)
            self.state = np.array([x, y, dets[2], s, dets[3], dets[4]])
            return
        c=self.c
        z_pred = np.array([x - (l * c) * np.cos(th), y - (l * c) * np.cos(th), th, l, w])
        H = np.array([[1, 0, l * c * np.sin(th), 0, c * np.cos(th), 0],
                      [0, 1, l * c * np.cos(th), 0, -c * np.cos(th), 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        v = det_vec - z_pred
        S = H.dot(self.P).dot(H.T)+self.R1
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        v[2] = wrapangle(v[2])
        self.state = self.state + v.dot(K.T)
        self.P = (self.P-K.dot(H).dot(self.P))



    def get_state(self):
        """
        Returns the current state estimate.
        """
        return self.state
    # def transform2meas(self):
    #     [x, y, th, s, l, w] = self.state
    #     c = self.c
    #     z_pred = np.array([z - (l / 4) * np.sin(th), x + (l / 4) * np.cos(th), th, l, w])
    #     H = np.array([[1, 0, -l * c * np.cos(th), 0, c * np.sin(th), 0],
    #                   [0, 1, -l * c * np.sin(th), 0, c * np.cos(th), 0],
    #                   [0, 0, 1, 0, 0, 0],
    #                   [0, 0, 0, 0, 1, 0],
    #                   [0, 0, 0, 0, 0, 1]])
    #     S = H @ self.P @ H.T
    #     return z_pred, S


def associate_detections_to_trackers(detections, trackers, conf_threshold=0.95):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    dist_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            dist_matrix[d, t] = mahab_dist(trk, det)
    matched_indices = linear_assignment(dist_matrix)
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    gate = chi2.ppf(conf_threshold, df=2)
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > gate:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=4, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.dt=0.1
        self.max_age = max_age  # number of frames without detections to delete track after
        self.min_hits = min_hits # number of hits to begin publishing track
        self.trackers = []
        self.frame_count = 0 # initialize number of frames so far in track to 0

    def update(self, dets, odometry, calib):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # print('dets in sort update to match to tracks', dets)
        self.frame_count += 1
        trks = []
        # print('current trk states')
        for t, trk in enumerate(self.trackers):
            print(trk.state)

            self.trackers[t].predict(odometry,calib)
            trks.append(self.trackers[t])
        to_del = []
        ret = []
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, 0.95)
        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d[0]],odometry)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # print('new  kalman car tracker object has been created')
            trk = KalmanCarTracker(dets[i],self.dt)
            self.trackers.append(trk)


        i = len(self.trackers)
        for trk in reversed(self.trackers):

            #appending matrices and vectors to trk to maintain history
            trk.history.append(trk.state)
            trk.Phist.append(trk.P)
            # z_pred, S = trk.transform2meas()
            # trk.Shist.append(S)
            # trk.zhist.append(z_pred)
            x =  trk.state[0]
            y = trk.state[1]

            # in_fov = (x-y >= 0) and (x+x >= 0)  #verify if a car is in camera fov (this case assumes 90 deg fov)
            in_fov = True
            #if a car is out of fov it wont be published

            #only track history is mantained for past 15 frames (this removes the oldest item in list)
            if len(trk.history) > 5:
                trk.history.pop(0)
                trk.Phist.pop(0)
                # trk.Shist.pop(0)
                # trk.zhist.pop(0)

            #if a track has gotten the min number of associated hits, then initialize it
            if trk.hit_streak >= self.min_hits:
                trk.initialize = True

            if ((trk.time_since_update < self.max_age) and ((trk.hits>=self.min_hits and trk.initialize) or self.frame_count <= self.min_hits)) and in_fov:
                ret.append(trk)

            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age) or not in_fov:
                self.trackers.pop(i)
        if (len(ret) > 0):
            return ret
            # return np.concatenate(ret)
        return []
        # return np.empty((0, 5))











