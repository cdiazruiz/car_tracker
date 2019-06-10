import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import SPF


def multiPeopleSPF(index,dic_X,dic_P,u,y,sigma_f)
    nx, nw, nv = 4,4,2
    na = nx+nw+nv 
    Wm = (sigma_f**2-na)/(sigma_f**2)
    Wc = ((sigma_f**2-na)/(sigma_f**2)+3-(sigma_f**2/na))
    
    fxw = lambda x,u,Xw: np.array([x[1], -u[0], x[3], -u[1]])*dt + x + Xw
    hxv = lambda x,Xv: np.hstack((np.sqrt(x[0]**2 + x[2]**2),math.atan2(x[2],x[0]))) + Xv
    
    Wm = np.hstack((Wm,np.ones((2*na))/(2*sigma_f**2)))
    Wc = np.hstack((Wc,np.ones((2*na))/(2*sigma_f**2)))

    for i in index:
        X = dic_X[i]
        P = dic_P[i]
        dic_X[i], dic_P[i]= SPF.SPF(X,P,Wm,Wc,u,y,sigma_f,nx,nw,nv,fxw,hxv)
        


