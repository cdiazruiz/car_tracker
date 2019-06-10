import numpy as np

def convert_vel(v_vec1,rot_mat,r_vec, w_vec):
    """input-v_vec= velocity vector [3x1]
    rotmat = 3x3 mat
    r_vec = position vector from A coord to B coord
    w_vec = angular veclocity of A coord system
    """
    # print('cross prod',np.cross(w_vec, r_vec))
    vel = v_vec1 + np.cross(w_vec, r_vec)
    v_vec=vel.dot(rot_mat.T)
    return v_vec