from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

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
    buf.shape = (w,h,4)
    buf = buf[:,:,0:3]
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = np.roll ( buf, 3, axis = 2 )
    return buf


def talker2():
        
    
        
    figure = plt.figure(dpi=10)
    plt.plot([1,2],[3,4])
    plt.axis([-30, 30, -30, 30])
    #plt.show()
    plotImg = fig2data (figure)
    
    print(plotImg)
    
    cv2.imshow("image",plotImg)
    time.sleep(100)
if __name__ == '__main__':
    
    talker2()
    