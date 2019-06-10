#!/usr/bin/env python
import rospy
from trackerapi.msg import TrackingResult
from trackerapi.msg import UnitTrackingResult

def talker():
    
    pub = rospy.Publisher('testTrackingInfo', TrackingResult, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        resultList = []
        for i in range(3):
            data = UnitTrackingResult()
            data.ID = i
            data.posX = 1
            data.posY = 1
            data.velX = 1
            data.velY = 1
            resultList.append(data)
        publishedResult = TrackingResult()
        publishedResult.result = resultList
        pub.publish(publishedResult)
        rate.sleep()

if __name__ == '__main__':
    try:
        
        talker()
    except rospy.ROSInterruptException:
        pass
