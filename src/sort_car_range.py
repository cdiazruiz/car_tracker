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

    det_vec = [det.z, det.x, det.rot_y, det.length, det.width]
    [z, x, th, s, l, w]=trk.state
    c = 0.25
    z_pred = np.array([z - (l*c) * np.sin(th), x + (l*c) * np.cos(th), th, l, w])
    H = np.array([[1, 0, -l * c * np.cos(th), 0, c * np.sin(th), 0],
                  [0, 1, -l * c * np.sin(th), 0, c * np.cos(th), 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    S = H@trk.P@H.T+trk.R1
    v=det_vec-z_pred
    v[2]=wrapangle(v[2])
    dist = v[0:2]@np.linalg.inv(S[0:2,0:2])@v[0:2].T
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

        self.initialize = False
        s=0  #initial speed is set to zero but will be updated with numerical differentiation
        self.c = 0.25  # 1/4 for estimate of length between center and back axle
        l_car = 4.7
        w_car = 1.9
        th=init_det.rot_y
        x = x_c - l_car * self.c * np.cos(th)
        z = z_c + l_car * self.c * np.sin(th)
        self.state = np.array([x, y, th, s, l, w])
        self. h = init_det.height
        self.y = init_det.y
        self.dT = dT
        self.type = init_det.type


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

        self.Q = np.identity(7)  # th s z_vel x_vel wy el ew
        self.c = 0.25 #1/4 for estimate of length between center and back axle
        # self.Q[0,0] = 0.02
        # self.Q[1,1] = 55
        # self.Q[2,2] = 35
        # self.Q[3,3] = 35
        # self.Q[4,4] = 0.08
        # self.Q[5,5] = 0.75
        # self.Q[6,6] = 0.75
        #process noise covariance matrix
        self.Q[0,0] = 0.030461174
        self.Q[1,1] = 1
        self.Q[2,2] = 0.0054783
        self.Q[3,3] = 0.00544783
        # self.Q[2, 2] = 0.54783
        # self.Q[3, 3] = 0.544783
        self.Q[4,4] = 0.030461
        self.Q[5,5] = 0.01
        self.Q[6,6] = 0.01
        self.Phist=[]
        self.zhist=[]
        self.Shist=[]
        # self.Phist.append(self.P)
        self.color=np.random.rand(1,3) #used only for display
        # self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 0, 1]])
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
        frame_diff=last_det.frame-pen_det.frame
        dt=frame_diff*self.dT
        vx=self.odo_hist[-2].vl
        vz=self.odo_hist[-2].vf
        wy=self.odo_hist[-2].wu
        px=vx*dt
        pz=vz*dt
        last = np.array([last_det.x, last_det.z, 1]).reshape((3,1))
        pen = np.array([pen_det.x, pen_det.z, 1]).reshape((3,1))
        th=-wy*dt
        Tr=np.array([[np.cos(th), -np.sin(th),px],[np.sin(th), np.cos(th), pz],[0, 0, 1]])
        iner_last=Tr@last
        dist = np.linalg.norm(iner_last-pen)
        s=dist/dt
        z_term=(last_det.z-pen_det.z)/dt + vz + pen_det.x*wy
        x_term=(last_det.x-pen_det.x)/dt + vx - pen_det.z*wy
        s=np.sqrt(z_term**2+x_term**2)
        th= math.atan2(-z_term,x_term)

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

        self.P = Ak@self.P@Ak.T + Wk@self.Q@Wk.T
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak= 0
        self.time_since_update += 1
        self.state = np.array([x, y, th, s, l, w])
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

    def jacobian_pred_meas(self,pos):

        [x,y,th,s,l,w]=self.state

        jr1 = np.array([(2 * x + (3 * l * np.cos(th)) / 2 - w * np.sin(th)) / (
                2 * ((x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)),
               (2 * y + w * np.cos(th) + (3 * l * np.sin(th)) / 2) / (
                       2 * ((x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                       y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), -(
                    2 * ((w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) * (
                    x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) - 2 * (
                            (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) * (
                            y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4)) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                    y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), 0, (
                       (3 * np.cos(th) * (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2)) / 2 + (
                       3 * np.sin(th) * (y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4)) / 2) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                    y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), (
                       np.cos(th) * (y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) - np.sin(th) * (
                       x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2)) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                    y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2))])

        jr2 = np.array([-((l * np.cos(th)) / 2 - 2 * x + w * np.sin(th)) / (
                2 * ((y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2)),
               (2 * y + w * np.cos(th) - (l * np.sin(th)) / 2) / (
                       2 * ((y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                       (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2)), -(
                    2 * ((l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) * (
                    y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) - 2 * (
                            (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) * (
                            (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2)) / (2 * (
                    (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                    (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2)), 0, -(
                    (np.sin(th) * (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4)) / 2 - (
                    np.cos(th) * ((l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2)) / 2) / (2 * (
                    (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                    (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2)), (
                       np.cos(th) * (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) + np.sin(th) * (
                       (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2)) / (2 * (
                    (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                    (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2))])

        jr3 = np.array([(2 * x - (l * np.cos(th)) / 2 + w * np.sin(th)) / (
                2 * ((x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)),
               -(w * np.cos(th) - 2 * y + (l * np.sin(th)) / 2) / (
                       2 * ((x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                       (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), (
                       2 * ((w * np.cos(th)) / 2 + (l * np.sin(th)) / 4) * (
                       x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) + 2 * (
                               (l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) * (
                               (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4)) / (2 * (
                    (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), 0, -(
                    (np.cos(th) * (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2)) / 2 - (
                    np.sin(th) * ((w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4)) / 2) / (2 * (
                    (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), (
                       np.sin(th) * (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) + np.cos(th) * (
                       (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4)) / (2 * (
                    (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2))])

        jr4 = np.array([(2 * x + (3 * l * np.cos(th)) / 2 + w * np.sin(th)) / (
                2 * ((x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)),
               (2 * y - w * np.cos(th) + (3 * l * np.sin(th)) / 2) / (
                       2 * ((x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                       y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), (
                       2 * ((w * np.cos(th)) / 2 - (3 * l * np.sin(th)) / 4) * (
                       x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) + 2 * (
                               (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) * (
                               y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4)) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), 0, (
                       (3 * np.cos(th) * (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2)) / 2 + (
                       3 * np.sin(th) * (y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4)) / 2) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), -(
                    np.cos(th) * (y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) - np.sin(th) * (
                    x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2)) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2))])

        jrm1 = np.array([(2 * x + (3 * l * np.cos(th)) / 2 - w * np.sin(th)) / (
                2 * ((x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)),
                (2 * y + w * np.cos(th) + (3 * l * np.sin(th)) / 2) / (
                        2 * ((x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                        y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), -(
                    2 * ((w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) * (
                    x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) - 2 * (
                            (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) * (
                            y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4)) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                    y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), 0, (
                        (3 * np.cos(th) * (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2)) / 2 + (
                        3 * np.sin(th) * (y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4)) / 2) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                    y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), (
                        np.cos(th) * (y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) - np.sin(th) * (
                        x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2)) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                    y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2))])

        jrm2 = np.array([-((l * np.cos(th)) / 2 - 2 * x + w * np.sin(th)) / (
                2 * ((y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2)),
                (2 * y + w * np.cos(th) - (l * np.sin(th)) / 2) / (
                        2 * ((y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                        (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2)), -(
                    2 * ((l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) * (
                    y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) - 2 * (
                            (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) * (
                            (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2)) / (2 * (
                    (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                    (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2)), 0, -(
                    (np.sin(th) * (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4)) / 2 - (
                    np.cos(th) * ((l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2)) / 2) / (2 * (
                    (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                    (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2)), (
                        np.cos(th) * (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) + np.sin(th) * (
                        (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2)) / (2 * (
                    (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                    (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2) ^ (1 / 2))])

        jrm3 = np.array([(2 * x - (l * np.cos(th)) / 2 + w * np.sin(th)) / (
                2 * ((x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)),
                -(w * np.cos(th) - 2 * y + (l * np.sin(th)) / 2) / (
                        2 * ((x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                        (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), (
                        2 * ((w * np.cos(th)) / 2 + (l * np.sin(th)) / 4) * (
                        x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) + 2 * (
                                (l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) * (
                                (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4)) / (2 * (
                    (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), 0, -(
                    (np.cos(th) * (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2)) / 2 - (
                    np.sin(th) * ((w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4)) / 2) / (2 * (
                    (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), (
                        np.sin(th) * (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) + np.cos(th) * (
                        (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4)) / (2 * (
                    (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2) ^ (1 / 2))])

        jrm4 = np.array([(2 * x + (3 * l * np.cos(th)) / 2 + w * np.sin(th)) / (
                2 * ((x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)),
                (2 * y - w * np.cos(th) + (3 * l * np.sin(th)) / 2) / (
                        2 * ((x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                        y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), (
                        2 * ((w * np.cos(th)) / 2 - (3 * l * np.sin(th)) / 4) * (
                        x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) + 2 * (
                                (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) * (
                                y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4)) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), 0, (
                        (3 * np.cos(th) * (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2)) / 2 + (
                        3 * np.sin(th) * (y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4)) / 2) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2)), -(
                    np.cos(th) * (y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) - np.sin(th) * (
                    x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2)) / (2 * (
                    (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                    y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2) ^ (1 / 2))])

        jb1 = np.array([-(y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) / (
                (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2),
               (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) / (
                       (x + (3 * l * np.cos(th)) / 4 - (w * np.sin(th)) / 2) ^ 2 + (
                       y + (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2), (
                       9 * l ^ 2 + 4 * w ^ 2 + 12 * l * x * np.cos(th) + 8 * w * y * np.cos(th) + 12 * l * y * np.sin(
                   th) - 8 * w * x * np.sin(th)) / (
                       (4 * x + 3 * l * np.cos(th) - 2 * w * np.sin(th)) ^ 2 + (
                       4 * y + 2 * w * np.cos(th) + 3 * l * np.sin(th)) ^ 2),
               0, -(6 * w + 12 * y * np.cos(th) - 12 * x * np.sin(th)) / (
                       9 * l ^ 2 + 24 * np.cos(th) * l * x + 24 * np.sin(th) * l * y + 4 * w ^ 2 - 16 * np.sin(
                   th) * w * x + 16 * np.cos(th) * w * y + 16 * x ^ 2 + 16 * y ^ 2),
               (6 * l + 8 * x * np.cos(th) + 8 * y * np.sin(th)) / (
                       9 * l ^ 2 + 24 * np.cos(th) * l * x + 24 * np.sin(th) * l * y + 4 * w ^ 2 - 16 * np.sin(
                   th) * w * x + 16 * np.cos(th) * w * y + 16 * x ^ 2 + 16 * y ^ 2)])

        jb2 = np.array([-(y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) / (
                (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2),
               -((l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) / (
                       (y + (w * np.cos(th)) / 2 - (l * np.sin(th)) / 4) ^ 2 + (
                       (l * np.cos(th)) / 4 - x + (w * np.sin(th)) / 2) ^ 2),
               (l ^ 2 + 4 * w ^ 2 - 4 * l * x * np.cos(th) + 8 * w * y * np.cos(th) - 4 * l * y * np.sin(
                   th) - 8 * w * x * np.sin(th)) / (
                       l ^ 2 - 8 * np.cos(th) * l * x - 8 * np.sin(th) * l * y + 4 * w ^ 2 - 16 * np.sin(
                   th) * w * x + 16 * np.cos(
                   th) * w * y + 16 * x ^ 2 + 16 * y ^ 2), 0, (2 * w + 4 * y * np.cos(th) - 4 * x * np.sin(th)) / (
                       l ^ 2 - 8 * np.cos(th) * l * x - 8 * np.sin(th) * l * y + 4 * w ^ 2 - 16 * np.sin(
                   th) * w * x + 16 * np.cos(
                   th) * w * y + 16 * x ^ 2 + 16 * y ^ 2), (8 * x * np.cos(th) - 2 * l + 8 * y * np.sin(th)) / (
                       l ^ 2 - 8 * np.cos(th) * l * x - 8 * np.sin(th) * l * y + 4 * w ^ 2 - 16 * np.sin(
                   th) * w * x + 16 * np.cos(
                   th) * w * y + 16 * x ^ 2 + 16 * y ^ 2)])

        jb3 = np.array([((w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) / (
                (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2),
               (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) / (
                       (x - (l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                       (w * np.cos(th)) / 2 - y + (l * np.sin(th)) / 4) ^ 2),
               (l ^ 2 + 4 * w ^ 2 - 4 * l * x * np.cos(th) - 8 * w * y * np.cos(th) - 4 * l * y * np.sin(
                   th) + 8 * w * x * np.sin(th)) / (
                       l ^ 2 - 8 * np.cos(th) * l * x - 8 * np.sin(th) * l * y + 4 * w ^ 2 + 16 * np.sin(
                   th) * w * x - 16 * np.cos(
                   th) * w * y + 16 * x ^ 2 + 16 * y ^ 2), 0, -(2 * w - 4 * y * np.cos(th) + 4 * x * np.sin(th)) / (
                       l ^ 2 - 8 * np.cos(th) * l * x - 8 * np.sin(th) * l * y + 4 * w ^ 2 + 16 * np.sin(
                   th) * w * x - 16 * np.cos(
                   th) * w * y + 16 * x ^ 2 + 16 * y ^ 2), -(8 * x * np.cos(th) - 2 * l + 8 * y * np.sin(th)) / (
                       l ^ 2 - 8 * np.cos(th) * l * x - 8 * np.sin(th) * l * y + 4 * w ^ 2 + 16 * np.sin(
                   th) * w * x - 16 * np.cos(
                   th) * w * y + 16 * x ^ 2 + 16 * y ^ 2)])

        jb4 = np.array([-(y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) / (
                (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2),
               (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) / (
                       (x + (3 * l * np.cos(th)) / 4 + (w * np.sin(th)) / 2) ^ 2 + (
                       y - (w * np.cos(th)) / 2 + (3 * l * np.sin(th)) / 4) ^ 2), (
                       9 * l ^ 2 + 4 * w ^ 2 + 12 * l * x * np.cos(th) - 8 * w * y * np.cos(th) + 12 * l * y * np.sin(
                   th) + 8 * w * x * np.sin(th)) / (
                       (4 * x + 3 * l * np.cos(th) + 2 * w * np.sin(th)) ^ 2 + (
                       4 * y - 2 * w * np.cos(th) + 3 * l * np.sin(th)) ^ 2),
               0, (6 * w - 12 * y * np.cos(th) + 12 * x * np.sin(th)) / (
                       9 * l ^ 2 + 24 * np.cos(th) * l * x + 24 * np.sin(th) * l * y + 4 * w ^ 2 + 16 * np.sin(
                   th) * w * x - 16 * np.cos(th) * w * y + 16 * x ^ 2 + 16 * y ^ 2),
               -(6 * l + 8 * x * np.cos(th) + 8 * y * np.sin(th)) / (
                       9 * l ^ 2 + 24 * np.cos(th) * l * x + 24 * np.sin(th) * l * y + 4 * w ^ 2 + 16 * np.sin(
                   th) * w * x - 16 * np.cos(th) * w * y + 16 * x ^ 2 + 16 * y ^ 2)])

        rminmax=self.which_rmin(pos)
        bminmax=self.occluding_angles(pos)
        rminid=rminmax[1]
        bminid=bminmax[1]
        bmaxid=bminmax[3]

        jr = np.vstack((jr1,jr2,jr3,jr4,jrm1,jrm2,jrm3,jrm4))[rminid,:]
        jbearings = np.vstack((jb1,jb2,jb3,jb4))

        H = np.vstack((jr,jbearings[bminid,:],jbearings[bmaxid,:]))

        v = np.array([rminmax[0]], bminmax[0], bminmax[2]).reshape(3,1)

        return H ,v

        #TODO: implement code

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

        if len(self.det_hist)>15:
            self.det_hist.pop(0)
            self.odo_hist.pop(0)
        if self.age<=4 and  1<=self.hits<=2: #happens for the f
            #initial speed is set to zero but will be updated with numerical differentiaon
            self.c = 0.25  # 1/4 for estimate of length between center and back axle
            #TODO: add differentiation based on centroid of pointcloud
            x_c=dets.x
            y_c=dets.y
            l_car=dets.length
            # th = dets.rot_y
            x = x_c - l_car * self.c * np.cos(th)
            z = z_c + l_car * self.c * np.sin(th)
            s, th = self.s_estimate()
            self.state = np.array([z, x, dets.rot_y, s, dets.length, dets.width])
            return

        # if not self.use_diff_speed:
        z_pred = np.array([z-(l/4)*np.sin(th), x +(l/4)*np.cos(th), th, l, w])
        # print('predicted measurement', z_pred)
        c=self.c

        S = H@self.P@H.T+self.R1
        K = self.P@H.T@np.linalg.inv(S)
        self.y = dets.y
        self.h = dets.height
        # v = np.array([dets.z, dets.x, dets.rot_y, dets.length, dets.width])-z_pred
        v[2] = wrapangle(v[2])
        self.state = self.state + v@K.T
        self.P = (self.P-K@H@self.P)


    def get_state(self):
        """
        Returns the current state estimate.
        """
        return self.state
    def transform2meas(self):
        [x, y, th, s, l, w] = self.state
        c = self.c
        z_pred = np.array([z - (l / 4) * np.sin(th), x + (l / 4) * np.cos(th), th, l, w])
        H = np.array([[1, 0, -l * c * np.cos(th), 0, c * np.sin(th), 0],
                      [0, 1, -l * c * np.sin(th), 0, c * np.cos(th), 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        S = H @ self.P @ H.T
        return z_pred, S


def associate_detections_to_trackers(detections, trackers, conf_threshold=0.95):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    dist_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        # print('det',det.__dict__)
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
        self.frame_count += 1
        trks = []
        for t, trk in enumerate(self.trackers):
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
            trk = KalmanCarTracker(dets[i])
            self.trackers.append(trk)


        i = len(self.trackers)
        for trk in reversed(self.trackers):

            #appending matrices and vectors to trk to maintain history
            trk.history.append(trk.state)
            trk.Phist.append(trk.P)
            z_pred, S = trk.transform2meas()
            trk.Shist.append(S)
            trk.zhist.append(z_pred)
            z =  trk.state[0]
            x = trk.state[1]

            in_fov = (z-x >= 0) and (z+x >= 0)  #verify if a car is in camera fov (this case assumes 90 deg fov)
            #if a car is out of fov it wont be published

            #only track history is mantained for past 15 frames (this removes the oldest item in list)
            if len(trk.history) > 15:
                trk.history.pop(0)
                trk.Phist.pop(0)
                trk.Shist.pop(0)
                trk.zhist.pop(0)

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


# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='SORT demo')
#     parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
#                         action='store_true')
#     args = parser.parse_args()
#     return args










