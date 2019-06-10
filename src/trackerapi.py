#!/usr/bin/env python

import time
import rospy
from std_msgs.msg import String
from novatel_gps_msgs.msg import NovatelVelocity
from novatel_gps_msgs.msg import NovatelCorrectedImuData
import PedestrainTracker


t_k_1_LDLS = -1
t_k_LDLS = -1
t_k_1_imu = -1
t_k_imu = -1
t_k_1_mrcnn = -1
t_k_mrcnn = -1
data_LDLS = None
data_imu = None
data_mrcnn = None
isLDLSValid = 0
isImuValid = 0
isMrcnnValid = 0
dt = -1


def callback_LDLS(data):
    global t_k_LDLS
    global data_LDLS
    global isLDLSValid
    isLDLSValid = 1
    data_LDLS = data
    header = data.header
    t_k_LDLS = header.stamp.to_sec()
    

def callback_imu(data):
    global t_k_imu
    global data_imu
    global isImuValid
    isImuValid = 1
    data_imu = data
    header = data.header
    t_k_imu = header.stamp.to_sec()
    
    
def callback_mrcnn(data):
    global t_k_mrcnn
    global data_mrcnn
    global isMrcnnValid
    isMrcnnValid = 1
    data_mrcnn = data
    header = data.header
    t_k_mrcnn = header.stamp.to_sec()

    

        
        
def listener(pub,tracker):

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    
    
    rospy.Subscriber('bestvel', NovatelVelocity, callback_vel)
    rospy.Subscriber('corrimudata', NovatelCorrectedImuData, callback_imu)
    global t_k_LDLS
    global data_LDLS
    global isLDLSValid
    global t_k_imu
    global data_imu
    global isImuValid
    global t_k_mrcnn
    global data_mrcnn
    global isMrcnnValid
    global t_k_1_LDLS
    global t_k_1_imu
    global t_k_1_mrcnn
    global dt
    # main loop
    while not rospy.is_shutdown():
        #sync: make sure all required data are collected
        if isLDLSValid != 0 and isImuValid != 0 and isMrcnnValid != 0:
            if t_k_1_vel != -1 and t_k_1_imu != -1 and t_k_1_mrcnn != -1:
                dt = ((t_k_LDLS-t_k_1_LDLS) + (t_k_imu-t_k_1_imu) + (t_k_mrcnn-t_k_1_mrcnn))/3
                # do stuff
                
                plotImg,image,pedestrainIndexs, pedestrainStates,pedestrains = tracker.track(
                    sefl,projection,detections,segResults,u,pedestrains)
                
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                #end doing stuff
            t_k_1_LDLS = t_k_LDLS
            t_k_1_imu = t_k_imu
            t_k_1_mrcnn = t_k_mrcnn
            isLDLSValid = 0
            isImuValid = 0
            isMrcnnValid = 0
        else:
            rospy.loginfo("Not Sync")
        time.sleep(0.0005)
        
    rospy.spin()

if __name__ == '__main__':
    try:
        tracker = PedestrainTracker()
        #image_pub = rospy.Publisher(pub_topic, Image, queue_size=QUEUE_SIZE)
        pub = rospy.Publisher('trackerInfo', String, queue_size=10)
        rospy.init_node('trackerapi', anonymous=True)
        listener(pub,tracker)
    except rospy.ROSInterruptException:
        pass
    
