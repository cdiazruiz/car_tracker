import rospy
import rosbag
import argparse
import os
import numpy as np


import rospy

from sensor_msgs.point_cloud2 import read_points, create_cloud_xyz32


from src.segmentation import LidarSegmentationResult, LidarSegmentation
from ldls_ros.msg import Segmentation, Pixor

def transform_cam2lid(det_arr):

    Rot1 = [0.6979, -0.7161, -0.0112, -0.0184, -0.0023, -0.9998, 0.7160, 0.6980, -0.0148]
    T1 = [-0.0876, -0.1172, -0.0710]
    Rot1 = np.array(Rot1)
    Rot1 = Rot1.reshape((3, -1))
    Rot2 = [0.6885, -0.7252, 0.0002, -0.0072, -0.0071, -0.9999, 0.7252, 0.6885, -0.0101]
    T2 = [-0.0864, -0.1423, -0.1942]
    Rot2 = np.array(Rot2)
    Rot2 = Rot2.reshape((3, -1))
    Rot = Rot1 * 3 / 4 + Rot2 / 4

    Transformation_lidar_cam1 = np.array(T1) * 3 / 4 + np.array(T2) / 4
    Transformation_lidar_cam1.shape = (-1, 1)
    Tr_lidar_cam1 = np.hstack((Rot, Transformation_lidar_cam1))
    Tr_lidar_cam1 = np.vstack((Tr_lidar_cam1, [0, 0, 0, 1]))


    Transformation_lidar_cam1 = np.array(T1) * 3 / 4 + np.array(T2) / 4
    Transformation_lidar_cam1.shape = (-1, 1)
    # Tr_lidar_cam1 = np.hstack((Rot, Transformation_lidar_cam1))
    # Tr_lidar_cam1 = np.vstack((Tr_lidar_cam1, [0, 0, 0, 1]))

    Rinv = Rot.T
    Tinv = -Rinv.dot(Transformation_lidar_cam1)
    # print('Tinv', Tinv)
    Tr_cam1_lidar = np.hstack((Rinv, Tinv))
    Tr_cam1_lidar = np.vstack((Tr_cam1_lidar, [0, 0, 0, 1]))

    n = det_arr.shape[0]
    points_c = np.hstack((det_arr[:, 0].reshape(-1, 1), np.zeros((n, 1)), det_arr[:, 1].reshape(-1, 1)))
    points_c = np.hstack((points_c, np.ones((n, 1))))
    points_c = Tr_cam1_lidar.dot(points_c.T).T

    return points_c[:,0:3]
def read_detections(det_path):
    '''
    Detections are in csv file where each row corresponds to a detection.
    If file is empty there were no detections for that frame.
    :param det_path:
    :return det_arr: n x 6 array it's in camera coordinates
                    each row =[x (right),  z (forward), theta, width, length, confidence]
    '''
    det_arr=np.genfromtxt(det_path, delimiter=',')
    return det_arr

def write_bag(input_path, output_path, det_path, mrcnn_results_topic, lidar_topic):
    """
    Reads an input rosbag, and writes an output bag including all input bag
    messages as well as Mask-RCNN results, written to the following topics:
    mask_rcnn/result: mask_rcnn_ros.Result
    mask_rcnn/visualization: Image
    Parameters
    ----------
    input_path: str
    output_path: str
    image_topic: str
    Returns
    -------
    """

    det_filenames = sorted(os.listdir(det_path))
    inbag = rosbag.Bag(input_path, 'r')
    outbag = rosbag.Bag(output_path, 'w')
    # lidar_list = []
    # lidar_headers = []
    # lidar_t = []
    # start = inbag.get_start_time()



    # Write all input messages to the output
    print("Reading messages...")
    for topic, msg, t in inbag.read_messages():
        outbag.write(topic, msg, t)


    # print("Running LDLS...")
    # lidarseg = LidarSegmentation(projection)
    i = 0
    for topic, msg, t in inbag.read_messages(topics=[mrcnn_results_topic]):
        if i % 50 == 0:
            print("Message %d..." % i)

        pdet_header = msg.header
        pdet_time = t
        det_array=read_detections(os.path.join(det_path, det_filenames[i]))
        if len(det_array)>0:
            det_array2lid= transfrom_cam2lid(det_array)
            det_array2lid_list=det_array2lid.tolist()
            det_list=det_array.tolist()
        else:
            det_list=[]
            det_array2lid_list=[]





        # Get the class IDs, names, header from the MRCNN message
        # class_ids = msg.class_ids
        # class_names = list(msg.class_names)
        # lidar = lidar_list[i]
        # header = lidar_headers[i]
        # ldls_res = lidarseg.run(lidar, detections, save_all=False)
        # lidar = lidar[ldls_res.in_camera_view, :]

        pixor_msg = Pixor()
        pixor_msg.header = pdet_header
        # ldls_msg.class_names = class_names
        # instance_ids = []
        # pc_msgs = []
        # # Get segmented point cloud for each object instance
        # labels = ldls_res.instance_labels()
        # class_labels = ldls_res.class_labels()
        # for inst in range(1, len(class_names) + 1):
        #     in_instance = labels == inst
        #     if np.any(in_instance):
        #         instance_ids.append(inst - 1)
        #         inst_points = lidar[in_instance, :]
        #         pc_msg = create_cloud_xyz32(header, inst_points)
        #         pc_msgs.append(pc_msg)
        pixor_msg.detections = det_list
        pixor_msg.object_points = create_cloud_xyz32(pdet_header, det_array2lid_list)
        outbag.write('/pixor/detections', pixor_msg, t)
        # outbag.write('/ldls/foreground', foreground_msg, t)
        i += 1
    inbag.close()
    outbag.close()


if __name__ == '__main__':
    mrcnn_results_topic = 'mask_rcnn/result'
    lidar_topic = '/velo3/pointcloud'
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile",
                        help="path to the bagfile to process")
    parser.add_argument('detections', help="path to the folder with all csv detection files")
    args = parser.parse_args()
    bag_path = args.bagfile
    det_path = args.detections
    if not os.path.exists(bag_path):
        raise IOError("Bag file '%s' not found" % bag_path)
    if not os.path.exists(det_path):
        raise IOError("Detections file '%s' not found" % det_path)
    out_name = bag_path.split('.bag')[0] + '_pixor.bag'
    write_bag(bag_path, out_name, det_path mrcnn_results_topic, lidar_topic)


