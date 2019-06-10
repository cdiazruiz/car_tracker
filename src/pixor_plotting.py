import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_velodyne_points(points_path):
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points

def read_detections(det_path):
    '''
    Detections are in csv file where each row corresponds to a detection.
    If file is empty there were no detections for that frame.
    :param det_path:
    :return det_arr: n x 7 array
                    each row =[x, y , z, theta, width, length, confidence]
    '''
    det_arr=np.genfromtxt(det_path, delimiter=',')
    return det_arr

def draw_rectangle(det_arr):
    '''

    :param det_arr:
    :return:
    '''
    #TODO: Add function to create corner points for rectangles for each detection


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("lidar", help="Path to lidar bin file")
    parser.add_argument("detections", help="Path to detections .csv file")
    args = parser.parse_args()

    #pointcloud in x y z of lidar
    pointcloud_lid = load_velodyne_points(args.lidar)

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

    Tr_lidar_cam0=np.array([[0.69555, -0.718375, -0.00835, -0.0873],
           [-0.0156, -0.0035, -0.999825, -0.123475],
           [0.7183, 0.695625, -0.013625, -0.1018],
           [0., 0., 0., 1.]])

    Transformation_lidar_cam1 = np.array(T1) * 3 / 4 + np.array(T2) / 4
    Transformation_lidar_cam1.shape = (-1, 1)
    Tr_lidar_cam1 = np.hstack((Rot, Transformation_lidar_cam1))
    Tr_lidar_cam1 = np.vstack((Tr_lidar_cam1, [0, 0, 0, 1]))

    Rinv = Rot.T
    Tinv = -Rinv.dot(Transformation_lidar_cam1)
    # print('Tinv', Tinv)
    Tr_cam1_lidar = np.hstack((Rinv, Tinv))
    Tr_cam1_lidar = np.vstack((Tr_cam1_lidar, [0, 0, 0, 1]))
     # = np.concatenate([pointcloud, np.ones((len(pointcloud), 1))], axis=1)
    pointcloud_lid = np.concatenate([pointcloud_lid, np.ones((len(pointcloud_lid), 1))], axis=1)
    pointcloud_cam = Tr_lidar_cam0.dot(pointcloud_lid.T).T
    #

    det_arr = read_detections(args.detections)
    n=det_arr.shape[0]
    # print(n)
    # print(det_arr[:,0:2])
    points_c = np.hstack((det_arr[:,0].reshape(-1,1),np.zeros((n,1)),det_arr[:,1].reshape(-1,1)))
    points_c = np.hstack((points_c, np.ones((n,1))))
    points_c = Tr_cam1_lidar.dot(points_c.T).T
    # # print('detections', det_arr[:,:3])
    x = pointcloud_cam[:,0]
    y = pointcloud_cam[:,1]
    z = pointcloud_cam[:,2]
    # print('points c', points_c)
    #

    car_x = det_arr[:,0]
    car_z = det_arr[:,1]
    # car_z = np.zeros(len(det_arr[:0]))
    plt.scatter(x,z,marker='.',s=1)
    plt.scatter(car_x, car_z, marker='o',s=4)
    # plt.xlim([0, 40])
    # plt.ylim([0,40])
    plt.gca().set_aspect('equal', adjustable='box')

    xl = pointcloud_lid[:, 0]
    yl = pointcloud_lid[:, 1]
    zl = pointcloud_lid[:, 2]
    car_x_l = points_c[:, 0]
    car_y_l = points_c[:,1]
    plt.figure()
    plt.scatter(xl, yl, marker='.', s=1)
    plt.scatter(car_x_l, car_y_l, marker='o', s=4)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()
