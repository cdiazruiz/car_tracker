# get bearing and depth measurement from image & bounding box
import numpy as np
import util


def projectLiddar(lidar, Tr):
    lidar = np.hstack((np.array(lidar),np.ones((len(lidar),1))))
    lidar = np.dot(Tr,lidar.T)
    return lidar.T

def measure_range(shape,box,mask,lidar,Proj):
    '''
    :param shape: image shape
    :param box: bbox
    :param mask: mask for an instance
    :param lidar: pointcloud for each segment
    :param Proj: projection object (contains camera intrinsics and transform from lidar to camera)
    :return: rmin, b_cen, b_max, b_min, [x,y] pos of min range ray
    '''

    proj_mat = Proj.projection_matrix
    AoV = np.arctan(proj_mat[0,2]/proj_mat[0,0])*2
    [m,n] = shape
    y1 = box[0]
    x1 = box[1]
    y2 = box[2]
    x2 = box[3]

    x = (x1+x2)/2
    y = (y1+y2)/2
    w = abs(x2-x1)
    h = abs(y2-y1)

    b_cen = -1 * np.arctan((x-(n/2))/((n/2)/np.tan(AoV/2)))
    b_max = -1 * np.arctan((x1 - (n/2)) / ((n/2) / np.tan(AoV/2)))
    b_min = -1 * np.arctan((x2 - (n/2)) / ((n/2) / np.tan(AoV/2)))

    cam_pcl = Proj.transform_lid2cam(lidar)
    rmin = min(np.sqrt(cam_pcl[:,1]**2+cam_pcl[:,2]**2))
    idx_rmin = np.argmin(cam_pcl[:, 1] ** 2 + cam_pcl[:, 2] ** 2)

    return np.array([rmin, b_min, b_max])
    #, np.array(cam_pcl[idx_rmin,2],cam_pcl[idx_rmin,1])]

def measure_3dbox():

    return np.array([x,y,th,l,w])

def pos_rmin(lidar, Proj):
    #returns the point in the segmented pcl in vel coordinates that yields rmin
    cam_pcl = Proj.transform_lid2cam(lidar)
    idx_rmin = np.argmin(cam_pcl[:, 1] ** 2 + cam_pcl[:, 2] ** 2)
    return [np.array([lidar[idx_rmin,0],lidar[idx_rmin,1]])]

