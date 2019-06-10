import numpy as np
import random
import time
import measurement_car
import util
from operator import itemgetter
from itertools import compress
from matplotlib import path


def agreement(corners, points, percentage = 0.3):
    '''
    determines if a certain % of points of a pointcloud are within a polygon (in this case rectangle)
    params: corners np.array() nx2 [x,y]
    :return: boolean
    '''
    pos = corners
    rectangle = path.Path(
        [(pos[0, 0], pos[0, 1]), (pos[1, 0], pos[1, 1]), (pos[2, 0], pos[2, 1]), (pos[3, 0], pos[3, 1])])
    inside = sum(rectangle.contains_points(pos[0:4, :]))
    match = (inside/points.shape[0]) >= percentage
    return match
def calc_corners(det):
    [x, y, th, l, w] = det
    cth=np.cos(th)
    sth=np.sin(th)
    c1 = np.array([x + (3 / 4) * l * cth - (1 / 2) * w * sth, y + (3 / 4) * l * sth + (1 / 2) * w * cth])
    c2 = np.array([x - (1 / 4) * l * cth - (1 / 2) * w * sth, y - (1 / 4) * l * sth + (1 / 2) * w * cth])
    c3 = np.array([x - (1 / 4) * l * cth + (1 / 2) * w * sth, y - (1 / 4) * l * sth - (1 / 2) * w * cth])
    c4 = np.array([x + (3 / 4) * l * cth + (1 / 2) * w * sth, y + (3 / 4) * l * sth - (1 / 2) * w * cth])
    pos =np.vstack((c1,c2,c3,c4))
    return pos

class ROSdetector:
    @staticmethod
    def getDetection(projection, detections, segResults):
        '''
        :param projection:
        :param detections:
        :param segResults:
        :return:
        '''
        min_pts_in_pcl=5
        shape = detections.shape
        vehicleList = [index for index, value in enumerate(detections.class_names) if value in ['car','truck','bus']]
        if len(vehicleList) == 0:
            t = time.time()
            return [t, [], [], []]
        boxes = list(itemgetter(*vehicleList)(detections.rois))
        masks = list(itemgetter(*vehicleList)(detections.masks))
        segResults = segResults.getPtCLoudsFromInstanceIds(vehicleList)
        if len(vehicleList) == 1 and len(segResults) >= min_pts_in_pcl:
            boxes = [boxes]
            masks = [masks]
            segResults = segResults
        measures = []
        masking =[]
        for i, box in enumerate(boxes):
            if len(segResults[i])>=min_pts_in_pcl:
                measures.append(measurement_car.measure(shape, box, masks[i], segResults[i], projection))
                masking.append(True)
            else:
                masking.append(False)
        t = time.time()
        returnDetection = [t, list(compress(measures,masking)), list(compress(boxes,masking)), list(compress(segResults,masking))]
        return returnDetection
    @staticmethod
    def getDetection_image_w_lidar(projection, image_detections, segResults, bb_3D):
        '''
        :param projection:
        :param detections:
        :param segResults:
        :param bb_3D:
        :return:
        '''
        # min_pts_in_pcl=5
        # shape = image_detections.shape
        # vehicleList = [index for index, value in enumerate(image_detections.class_names) if value in ['car','truck','bus']]
        # if len(vehicleList) == 0:
        #     t = time.time()
        #     return [t, [], [], []]
        # boxes = list(itemgetter(*vehicleList)(image_detections.rois))
        # masks = list(itemgetter(*vehicleList)(image_detections.masks))
        # segResults = segResults.getPtCLoudsFromInstanceIds(vehicleList)
        # segResults_cam=[]
        # if len(vehicleList) == 1 and len(segResults) >= min_pts_in_pcl:
        #     boxes = [boxes]
        #     masks = [masks]
        #     segResults = segResults
        # measures = []
        # masking =[]
        # # for i, box in enumerate(boxes):
        # #     for ii, bb in enumerate(bb_3D):
        # #         if len(segResults[i]) >= min_pts_in_pcl:
        # #             segmentedpcl_cam=projection.transform_lid2cam(segResults[i])
        # #             match = agreement(calc_corners(bb), np.hstack(segmentdpcl[0,:]))
        # #             measures.append(bb)
        # #
        #         masking.append(True)
        #     else:
        #         masking.append(False)
        # t = time.time()
        # returnDetection = [t, [], list(compress(boxes,masking)), list(compress(segResults_cam,masking))]
        cut_off_score=0.1
        # print('bb_3D', bb_3D)
        # print('bb_3D.size',bb_3D.size)
        if bb_3D.size==0:
            return []
        elif bb_3D.size>6:
            filteredDetections = np.hstack([bb_3D[:,1].reshape(-1,1),-bb_3D[:,0].reshape(-1,1),
                                            -bb_3D[:,2].reshape(-1,1)-np.pi/2,bb_3D[:,4].reshape(-1,1),
                                            bb_3D[:,3].reshape(-1,1),bb_3D[:,5].reshape(-1,1)])
            # print('filtDet', filteredDetections)
            filteredDetections = filteredDetections[filteredDetections[:,-1]>=cut_off_score,:].tolist()
        else:
            if bb_3D[0,-1]>=cut_off_score:
                # print(bb_3D[2,0])
                bb_3D=bb_3D.flatten()
                filteredDetections = [np.array([bb_3D[1], -bb_3D[0], -bb_3D[2] - np.pi/2, bb_3D[4], bb_3D[3],bb_3D[5]])]
            else:
                filteredDetections=[]

        # for i, bb in enumerate(bb_3D):
        #     for segmentedpcl in returnDetection[3]:
        #         match=agreement(calc_corners(bb),projection.transform_lid2cam(segmentedpcl))
        #         if match:
        #             filteredDetections.append(bb)
        #             break
        return filteredDetections
