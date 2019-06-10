#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic
import time
import rospy
from std_msgs.msg import String
from novatel_gps_msgs.msg import NovatelVelocity
from novatel_gps_msgs.msg import NovatelCorrectedImuData
t_k_1_vel = -1
t_k_vel = -1
t_k_1_imu = -1
t_k_imu = -1
data_vel = None
data_imu = None
isVelValid = 0
isImuValid = 0
dt = -1


def callback_vel(data):
    global t_k_vel
    global data_vel
    global isVelValid
    isVelValid = 1
    data_vel = data
    header = data.header
    t_k_vel = header.stamp.to_sec()
    

def callback_imu(data):
    global t_k_imu
    global data_imu
    global isImuValid
    isImuValid = 1
    data_imu = data
    header = data.header
    t_k_imu = header.stamp.to_sec()
    

    

        
        
def listener(pub):

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    
    
    rospy.Subscriber('bestvel', NovatelVelocity, callback_vel)
    rospy.Subscriber('corrimudata', NovatelCorrectedImuData, callback_imu)
    global t_k_vel
    global data_vel
    global isVelValid
    global t_k_imu
    global data_imu
    global isImuValid
    global dt
    global t_k_1_vel
    global t_k_1_imu
    
    # spin() simply keeps python from exiting until this node is stopped
    while not rospy.is_shutdown():
        if isVelValid != 0 and isImuValid != 0:
            if t_k_1_vel != -1 and t_k_1_imu != -1:
                dt = ((t_k_vel-t_k_1_vel) + (t_k_imu-t_k_1_imu))/2
            
            rospy.loginfo(dt)
            pub.publish("dt1: " + str((t_k_vel-t_k_1_vel)) + " dt2: " + str((t_k_imu-t_k_1_imu)) + " dt: " + str(dt))
            t_k_1_vel = t_k_vel
            t_k_1_imu = t_k_imu
            isVelValid = 0
            isImuValid = 0
        else:
            rospy.loginfo("Not Sync")
        time.sleep(0.0005)
        
    rospy.spin()

if __name__ == '__main__':
    try:
        pub = rospy.Publisher('trackerInfo', String, queue_size=10)
        rospy.init_node('listener', anonymous=True)
        listener(pub)
    except rospy.ROSInterruptException:
        pass
    
