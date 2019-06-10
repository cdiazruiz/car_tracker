# get bearing and depth measurement from image & bounding box
import numpy as np


def projectLiddar(lidar, Tr):
    lidar = np.hstack((np.array(lidar),np.ones((len(lidar),1))))
    print(lidar.T)
    lidar = np.dot(Tr,lidar.T)
    print(lidar.T)
    return lidar.T



def measure(shape,box,mask):
    '''
    c = 
    [m,n,z] = img.shape
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]    
    theta = np.arctan((x-(n/2))/(n/2)*np.tan(AoV/2))
    Y = y-h/2
    z = -c/Y
    '''   
    
    
    
    AoV = 60.0/180.0*np.pi
    #pixel2m = 0.0001  #camera param to be calibrated
    #pin2plane= 7                    #camera param to be calibrated
    H_human = 1.8                   #height of human [m]
    [m,n] = shape
    y1 = box[0]
    x1 = box[1]
    y2 = box[2]
    x2 = box[3]
    x = (x1+x2)/2
    y = (y1+y2)/2
    w = abs(x2-x1)
    h = abs(y2-y1)
    '''
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    '''
    #theta = x/n*AoV-(AoV/2)
    theta = np.arctan((x-(n/2))/((n/2)/np.tan(AoV/2)))
    #h_m = h*pixel2m
    #r = H_human/h_m*pin2plane*0.001
    r = (m/h)*((H_human/2.0)/(np.tan(AoV/2)))
    
    return [np.array([r,theta]),mask]



def measure2(shape,box,mask,lidar,Tr,Proj):
    '''
    c = 
    [m,n,z] = img.shape
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]    
    theta = np.arctan((x-(n/2))/(n/2)*np.tan(AoV/2))
    Y = y-h/2
    z = -c/Y
    '''   
    

    AoV = np.arctan(Proj[0,2]/Proj[0,0])*2
    #pixel2m = 0.0001  #camera param to be calibrated
    #pin2plane= 7                    #camera param to be calibrated
    H_human = 1.6                  #height of human [m]
    [m,n] = shape
    y1 = box[0]
    x1 = box[1]
    y2 = box[2]
    x2 = box[3]
    x = (x1+x2)/2
    y = (y1+y2)/2
    w = abs(x2-x1)
    h = abs(y2-y1)
    '''
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    '''
    #theta = x/n*AoV-(AoV/2)
    theta = -1*np.arctan((x-(n/2))/((n/2)/np.tan(AoV/2)))
    r_camera = (m/h)*((H_human/2.0)/(np.tan(AoV/2)))
    #h_m = h*pixel2m
    #r = H_human/h_m*pin2plane*0.001
    print("===================POS CHECK ====================")
    print("lidar:")
    print(lidar)
    if len(lidar) == 0:
        return [np.array([theta]),r_camera,mask]
    elif len(lidar) == 1:
        if len(lidar[0]) == 0:
            return [np.array([theta]),r_camera,mask]
    
    pos = np.nansum(np.array(lidar),axis=0)/len(lidar)
    pos = np.hstack((pos,1))
    print("pos:")
    print(pos)
    pos.shape = (4,1)
    pos = np.dot(Tr,pos).T[0]
    print("overall pose:")
    print(pos)
    print("camera based range estimation")
    print(r_camera)
    
    r = (pos[0]**2+pos[2]**2)**0.5
    #r = float('nan')
    print("lidar based range estimation")
    print(r)
    if np.isnan(r):
        return [np.array([theta]),r_camera,mask]
    else:
        return [np.array([r,theta]),r_camera,mask]
    
    
