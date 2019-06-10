#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np


def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.array(fig.canvas.renderer.buffer_rgba())
    buf.shape = ( w,h,4 )
    buf = buf[:,:,0:3]
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = np.roll ( buf, 3, axis = 2 )
    return buf


def talker2(bridge):
        
    pub = rospy.Publisher('testPlot', Image, queue_size=10)
    rospy.init_node('testPlotNode', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world2 %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        figure = plt.figure(dpi=100)
        plt.plot([1,2],[3,4])
        plt.axis([-30, 30, -30, 30])
        figure.canvas.draw()
        plotImg = fig2data (figure)
        print(plotImg.shape)
        
        pub.publish(bridge.cv2_to_imgmsg(plotImg, "rgb8"))
        plt.close()
        rate.sleep()

if __name__ == '__main__':
    try:
        bridge = CvBridge()
        talker2(bridge)
    except rospy.ROSInterruptException:
        pass