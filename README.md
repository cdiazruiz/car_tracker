# car_tracker
ROS Package for vehicle tracking using pointcloud detector

pip install -r requirements.txt in your virtualenv

catkin_make along with ldls library. 

To run tracker:

In one terminal: roscore

Second terminal: rviz

Third terminal: rosbag play <.bag>

Fourth terminal:

source devel/setup.bash

rosrun trackerapi trackerapi3.py
