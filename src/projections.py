"""
util.py
Brian Wang

Utilities for loading in lidar and image data.

"""
import numpy as np
#from skimage.io import imread

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class Projection(object):
    """
    Class for projecting lidar points into a 2D image frame.
    
    Project points using the Projection.project() method.
    
    Attributes
    ----------
    T: numpy.ndarray
        Transformation matrix. 4 by 4
        Transforms 3D homogeneous coordinate lidar points to 3D homogeneous
        cooordinate points in the camera fram.e
    P: numpy.ndarray
        Projection matrix. 3 by 4.
        Project a 3D point (x,y,z) to 2D image coordinates by appending a 1,
        for homogeneous coordinates, and then multiplying by P.
        
        R = P * [x y z 1]'
        
        Then, the image row coordinate is R[0]/R[2],
        and the column coordinate is R[1]/R[2]
        (i.e. divide the first and second dimensions by the third dimension)
        
    """


    def __init__(self, Tr, Tr_inv, P):
        self.transformation_matrix = Tr
        self.inverse_transformation =Tr_inv
        self.projection_matrix = P

    def project(self, points, remove_behind=True):
        """
        Project points from the Velodyne coordinate frame to image frame
        pixel coordinates.
        
        Parameters
        ----------
        points: numpy.ndarray
            n by 3 numpy array.
            Each row represents a 3D lidar point, as [x, y, z]
        remove_behind: bool
            If True, projects all lidar points that are behind the camera
            (checked as x <= 0) to NaN

        Returns
        -------
        numpy.ndarray
            n by 2 array.
            Each row represents a point projected to 2D camera coordinates
            as [row, col]

        """
        n = points.shape[0]
        d = points.shape[1]
        Tr = self.transformation_matrix
        P = self.projection_matrix
        if d == 3:
            # Append 1 for homogenous coordinates
            points = np.concatenate([points, np.ones((n, 1))], axis=1)
        projected = (P.dot(Tr).dot(points.T)).T

        # normalize by dividing first and second dimensions by third dimension
        projected = np.column_stack(
            [projected[:, 0] / projected[:, 2],
             projected[:, 1] / projected[:, 2]])

        if remove_behind:
            behind = points[:,0] <= 0
            projected[behind,:] = np.nan

        return projected

    def transform_cam2lid3D(self, points):
        points = np.reshape(points, (-1, 3))
        points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        return self.inverse_transformation.dot(points.T).T

    def transform_cam2lid2D(self, points):
        '''
        points nx2 np.array (x,z)
        points are provided in camera coord system : (z) forward and right (x)
        z=forward, x=right, y=down for camera
        :param points:
        :return: np.array(x,y,z) in lidar coords x=forward, y = left, z = up
        '''

        points = np.reshape(points, (-1, 2))
        n = points.shape[0]
        # print(points[:, 0].reshape(-1, 1),np.zeros((n, 1)), points[:, 1].reshape(-1, 1))
        points_c = np.hstack((points[:, 0].reshape(-1, 1), np.zeros((n, 1)), points[:, 1].reshape(-1, 1)))
        points_c = np.hstack((points_c, np.ones((n, 1))))
        return self.inverse_transformation.dot(points_c.T).T
    def transform_lid2cam(self, points):
        points= np.reshape(points, (-1, 3))
        points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        # print('points', points)
        return self.transformation_matrix.dot(points.T).T






