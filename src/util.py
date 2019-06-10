
from projections import Projection
import numpy as np
import ros_numpy

def rosROIstoROIs(rosROIs):
    ROIS = []
    for rosROI in rosROIs:
        x_offset = rosROI.x_offset
        y_offset = rosROI.y_offset
        height = rosROI.height
        width = rosROI.width
        ROIS.append([y_offset,x_offset,y_offset+height,x_offset+width])
    return ROIS


def rosImgtoImg(bridge,rosImg):
    cv_image = bridge.imgmsg_to_cv2(rosImg, "bgr8")
    return cv_image

def rosImgstoImgs(bridge,rosImgs):
    imgs = []
    for rosImg in rosImgs:
        imgs.append(rosImgtoImg(bridge,rosImg))
    return imgs



def rosPtCloudtoPtCloud(rosPtCloud):
    return ros_numpy.point_cloud2.pointcloud2_to_xyz_array(rosPtCloud,remove_nans = False)

def rosPtCloudstoPtClouds(rosPtClouds):
    ptClouds = []
    for rosPtCloud in rosPtClouds:
        ptClouds.append(rosPtCloudtoPtCloud(rosPtCloud))
    return ptClouds

def getProjection():
    Rot1 = [0.6979,-0.7161,-0.0112,-0.0184,-0.0023,-0.9998,0.7160,0.6980,-0.0148]
    T1 = [-0.0876,-0.1172,-0.0710]
    Rot1 = np.array(Rot1)
    Rot1 = Rot1.reshape((3,-1))
    Rot2 = [0.6885,-0.7252,0.0002,-0.0072,-0.0071,-0.9999,0.7252,0.6885,-0.0101]
    T2 = [-0.0864,-0.1423,-0.1942]
    Rot2 = np.array(Rot2)
    Rot2 = Rot2.reshape((3,-1))
    Rot = Rot1*3/4+Rot2/4
    Transformation_lidar_cam1 = np.array(T1)*3/4+np.array(T2)/4
    Transformation_lidar_cam1.shape = (-1,1)
    Tr_lidar_cam1 = np.hstack((Rot,Transformation_lidar_cam1))
    Tr_lidar_cam1 = np.vstack((Tr_lidar_cam1,[0,0,0,1]))

    Rinv = Rot.T
    Tinv = -Rinv.dot(Transformation_lidar_cam1)
    Tr_cam1_lidar = np.hstack((Rinv, Tinv))
    Tr_cam1_lidar = np.vstack((Tr_cam1_lidar, [0, 0, 0, 1]))


    cx =  960.00000
    cy = 604.00000
    distortion= [-2.89514064788818e-1, 8.77557247877121e-2, 0.000000000000000]
    fx= 1808.7218
    fy=1812.3969
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    Rot_cam1 = np.eye((3))
    Tr_cam1 = np.zeros((3,1))
    R_tr_cam1 = np.hstack((Rot_cam1,Tr_cam1))
    Proj_cam1 = np.dot(K,R_tr_cam1)
    return Projection(Tr_lidar_cam1, Tr_cam1_lidar, Proj_cam1)
        