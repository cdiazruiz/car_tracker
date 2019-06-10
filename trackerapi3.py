#!/usr/bin/env python
import copy
import time
import rospy
import numpy as np
from src import PedestrainTracker
from src.detections import MaskRCNNDetections
from src.segResults import segResults
#from segResults import segResults
from src import projections
from src import util
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import matplotlib.pyplot as plt
# ============== messages ===================
from std_msgs.msg import String, Header
from novatel_gps_msgs.msg import NovatelVelocity
from novatel_gps_msgs.msg import NovatelCorrectedImuData
from novatel_gps_msgs.msg import Inspva
from mask_rcnn_ros.msg import Result
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from ldls_ros.msg import Segmentation, Pixor
from trackerapi.msg import TrackingResult
from trackerapi.msg import UnitTrackingResult
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
# from
# ============================================

#carlos added
from src import sort_car

##carlos added these to test line fitting
from src import detector_car
# from src import measurement_car
from operator import itemgetter
from src import fit_lines

class trackerAPI:

    def __init__(self):
        # ============== define instance variables to store data ==============

        self.t_k_LDLS = -1
        self.t_k_imu = -1
        self.t_k_mrcnn = -1
        self.t_k_inspva = -1
        self.t_k_3Ddetect = -1
        self.t_k = -1
        self.t_k_1 = -1
        self.data_LDLS = None
        self.data_imu = None
        self.data_mrcnn = None
        self.data_inspva = None
        self.data_3Ddetect = None
        self.isInspvaValid = 0
        self.isDataValid = 0
        self.dt = -1

        self.num=0
        self.azi_list = []
        self.ins_time = []
        # Assuming time is synced so we don't really care about the time of raw image and lidar
        self.rawImg = None
        self.rawLidar = None
        # self.tracker = PedestrainTracker.PedestrainTracker()
        self.tracker = sort_car.Sort()
        self.bridge = CvBridge()
        # =============== define publishers =====================
        self.testPub = rospy.Publisher('trackerapi/Info', String, queue_size=10)
        self.debugPub = rospy.Publisher('trackerapi/DebugInfo', String, queue_size=10)
        self.imgPub = rospy.Publisher('trackerapi/Img', Image, queue_size=10)
        self.resultPub = rospy.Publisher('trackerapi/Result', TrackingResult, queue_size=10)
        self.visualizePub = rospy.Publisher('trackerapi/Visualizing', MarkerArray, queue_size=10)
        
        # =============== run the node ==========================
        rospy.init_node('trackerapi', anonymous=False)
        self.listener()


    def callback_inspva(self,data):
        self.isInspvaValid = 1
        self.data_inspva = data
        header = data.header
        self.t_k_inspva = header.stamp.to_sec()
        # self.Nvel_list.append(data.north_velocity)
        # self.Evel_list.append(data.east_velocity)
        self.azi_list.append(data.azimuth*np.pi/180)
        self.ins_time.append(header.stamp.to_sec())

    def callback_3Ddetect(self,data):
        self.is3DdetectValid =1
        self. data_3Ddetect = data
        header = data.header
        self.t_k_3Ddetect= header.stamp.to_sec()


    def callback_gotAllInfo(self,data_ldls,data_mrcnn,data_img,data_lidar,data_3Ddetector):
        self.t_k_mrcnn = data_mrcnn.header.stamp.to_sec()
        self.t_k_LDLS = data_ldls.header.stamp.to_sec()
        self.data_LDLS = data_ldls
        self.rawLidar = data_lidar
        self.rawImg = data_img
        self.data_mrcnn = data_mrcnn
        self.data_3Ddetect = data_3Ddetector
        self.t_k  = max(self.t_k_LDLS,self.t_k_mrcnn)
        self.isDataValid = 1

    def getDetections(self):
        shape = [self.rawImg.height,self.rawImg.height]
        rois = util.rosROIstoROIs(self.data_mrcnn.boxes)
        class_ids = self.data_mrcnn.class_ids
        scores = self.data_mrcnn.scores
        masks = util.rosImgstoImgs(self.bridge,self.data_mrcnn.masks)
        return MaskRCNNDetections(shape,rois,masks,class_ids,scores)

    def getSegResults(self):
        instanceIDs = self.data_LDLS.instance_ids
        segResultPts = util.rosPtCloudstoPtClouds(self.data_LDLS.object_points)
        return segResults(instanceIDs,segResultPts)

    def get3Ddetections(self):
        detvector = self.data_3Ddetect.detections
        if len(detvector)==1:
            return np.array([])
        else:
            det_arr=np.asarray(detvector).reshape(-1,6)
            return det_arr

    def getEgoStatus(self):
        north_v = self.data_inspva.north_velocity
        east_v = self.data_inspva.east_velocity
        phi = self.data_inspva.azimuth*np.pi/180
        #self.azi_list.append(phi)
        vx = np.cos(phi) * north_v + np.sin(phi) * east_v
        vy = np.sin(phi) * north_v - np.cos(phi) * east_v
        '''
        second degree backward finite difference for yaw rate
        '''
        #azimuth_list = self.azi_list
        #self.debugPub.publish(str(azimuth_list))

        #TODO: guard against division by zero
        # if len(self.azi_list)>=3:
        #     wz=-1*(3*self.azi_list[-1]-4*self.azi_list[-2]+self.azi_list[-3])/(2*(self.ins_time[-1]-self.ins_time[-2]))
        # else:
        #     wz=0

        wz = 0
        if len(self.azi_list) >= 3:
            h = ((self.ins_time[-1] - self.ins_time[-2]) + (self.ins_time[-2] - self.ins_time[-3])) / 2
            if abs(h - 0) > 1E-6:
                wz = -1 * (3 * self.azi_list[-1] - 4 * self.azi_list[-2] + self.azi_list[-3]) / (2 * h)
        self.azi_list=[]
        self.ins_time=[]
        # vx=0
        # vy=0
        # wz=0

        return [vx, vy, wz]


    def publishTrackingVisualization(self, car_tracks, projection,t):

        resultList = []
        marker= Marker()
        marker.action = 3
        header = Header()
        header.stamp = rospy.Time.from_sec(t)
        header.frame_id ="/velodyne"
        marker.header = header
        resultList.append(marker)
        print('car track objects',car_tracks)
        for car in car_tracks:
            print('publising car')
            marker = Marker()
            state = car.state[0:2]
            positions = projection.transform_cam2lid2D(np.array([-state[1],state[0]]))
            positions = positions.flatten()
            marker.id = car.id
            marker.type = 2
            marker.action = 0;
            marker.pose.position.x = positions[0]
            marker.pose.position.y = positions[1]
            marker.pose.position.z = 0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.5*4;
            marker.scale.y = 0.5*4;
            marker.scale.z = 0.5;
            marker.color.a = 0.7; #Don't forget to set the alpha!
            # print('color', car.color)

            marker.color.r = car.color.flatten()[0];
            marker.color.g = car.color.flatten()[1];
            marker.color.b = car.color.flatten()[2];
            header = Header()
            header.stamp = rospy.Time.from_sec(t);
            header.frame_id = "/velodyne"
            marker.header = header
            resultList.append(marker)

        publishedResult = MarkerArray()
        publishedResult.markers = resultList
        self.visualizePub.publish(publishedResult)


    def publishTrackingResult(self,pedestrianIndexs, pedestrianStates,t):
        resultList = []
        for pedestrianIndex in pedestrianIndexs:
            data = UnitTrackingResult()
            data.ID = pedestrianIndex
            state = pedestrianStates[pedestrianIndex]
            data.posX = state[0]
            data.posY = state[2]
            data.velX = state[1]
            data.velY = state[3]
            resultList.append(data)

        publishedResult = TrackingResult()
        header = Header()
        header.stamp = rospy.Time.from_sec(t);
        publishedResult.header = header
        publishedResult.result = resultList
        self.resultPub.publish(publishedResult)


    def listener(self):

        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.


        # ============== Unpack instance variables ==============



        # =======================================================

        # =================== Set up subscribers ================
        #rospy.Subscriber('bestvel', NovatelVelocity, callback_vel)

        queue_size = 50

        tss = ApproximateTimeSynchronizer([Subscriber('ldls/segmentation', Segmentation),
        Subscriber('mask_rcnn/result', Result),
        Subscriber('raw_image', Image),
        Subscriber('velo/pointcloud', PointCloud2),Subscriber('pixor/detections',Pixor)],queue_size,0.1)


        tss.registerCallback(self.callback_gotAllInfo)

        rospy.Subscriber('inspva', Inspva, self.callback_inspva)
        Subscriber('inspva', Inspva)
        # =======================================================
        #generate projection


        projection = util.getProjection()
        # sort_tracker= sort_car.Sort()


        # ======================= main loop =====================
        while not rospy.is_shutdown():
            #sync: make sure all required data are collected
            #if self.isLDLSValid != 0 and self.isInspvaValid != 0 and self.isMrcnnValid != 0 and self.isRawImgValid == 1 and self.isRawLidarValid == 1:
            if self.isDataValid == 1:


                if self.t_k_1 != -1 and self.t_k != -1:
                    self.dt = self.t_k-self.t_k_1
                    self.tracker.dt=self.dt
                    # =========== decode ========================
                    print('########### Frame Num ############', self.num)
                    self.num+=1
                    image = copy.deepcopy(self.rawImg)
                    #image = self.rawImg
                    lidar = copy.deepcopy(self.rawLidar)

                    image_detections = self.getDetections()
                    pcl_detections = self.get3Ddetections()
                    # print('pixor detections', pcl_detections)
                    segResults = self.getSegResults()
                    u = self.getEgoStatus()

                    image = self.bridge.imgmsg_to_cv2(image, "rgb8")
                    lidar = util.rosPtCloudtoPtCloud(lidar)
                    # print('raw bb box detections', pcl_detections)

                    detections2meas = detector_car.ROSdetector.getDetection_image_w_lidar(projection, image_detections,
                                                                                     segResults, pcl_detections)
                    # print('detections2meas', detections2meas)
                    print('number of detections', len(detections2meas))
                    print('odometry', u)
                    pedestrianStates=[]


                    tracks = self.tracker.update(detections2meas, u, projection)
                    # print('number of kalman trackers active', len(self.tracker.trackers))
                    # print('kalman objecs', self.tracker.trackers)
                    # #TODO: fix rest of code after this
                    # if len(detections[1])>0:
                    #     for count, segmented_pcl in enumerate(detections[3]):
                    #         # print('segmented_pcl', segmented_pcl)
                    #         if len(segmented_pcl)>0:
                    #
                    #             if segmented_pcl.shape[0]>=5:
                    #                 # print('segmented_pcl', segmented_pcl)
                    #                 pedestrianStates.append(measurement_car.pos_rmin(segmented_pcl, projection))
                    #                 # pedestrianStates.append(fit_lines.lines(segmented_pcl[:,0],segmented_pcl[:,1]))
                    #                 num_pcls+=1
                    # [projectedLidar,image,pedestrianIndexs, pedestrianStates,pedestrians] = self.tracker.track(
                    #     projection,detections,segResults,u,image,lidar,self.dt)


                    #publish
                    # self.publishTrackingResult(pedestrianIndexs, pedestrianStates,self.t_k)
                    self.publishTrackingVisualization(tracks, projection,self.t_k)
                    # self.imgPub.publish(self.bridge.cv2_to_imgmsg(image, "rgb8"))
                    # self.debugPub.publish(str([round(u[0],2),round(u[1],5)]))
                    # self.testPub.publish("-ldls: " + str(self.t_k_LDLS) + " -mrcnn: " +
                    #             str(self.t_k_mrcnn) + " -inspva: " + str(self.t_k_inspva) + " -tk: " + str(self.t_k))



                    #end doing stuff

                #Store data

                '''
                self.t_k_1_LDLS = self.t_k_LDLS
                self.t_k_1_imu = self.t_k_imu
                self.t_k_1_mrcnn = self.t_k_mrcnn
                self.t_k_1_inspva = self.t_k_inspva
                '''
                self.isInspvaValid = 0
                self.isDataValid = 0

                self.t_k_1 = self.t_k

                #rospy.loginfo("Not Sync -ldls %d -imu %d -mrcnn %d -inspva %d", self.isLDLSValid,
#self.isImuValid,self.isMrcnnValid,self.isInspvaValid)




        rospy.spin()




if __name__ == '__main__':
    try:
        trackerAPI()
    except rospy.ROSInterruptException:
        pass
