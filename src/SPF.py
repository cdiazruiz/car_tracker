import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

def f(X,u,dt = 1/30):
    #x: [x,dx,y,dy]
    #u: [v,w]
    x = X[0]
    y = X[2]
    v = u[0]
    w = u[1]
    dx_ego = v
    dy_ego = 0
    dx = X[1]-dx_ego
    dy = X[3]-dy_ego
    dX = np.array([dx-y*w,-2*dy*w-x*(w**2),dy+x*w,2*dx*w-y*(w**2)])
    X_k = dX*dt + X

    return X_k


def multiPeopleSPF(index,dic_X,dic_P,u,y,sigma_f,dt):
## y: dictionary, and 2d array like ([[r theta]])
   
    if len(index)==0:
        return index, dic_X, dic_P,{}
    nx, nw, nv = 4,4,2
    na = nx+nw+nv 
    Wm = (sigma_f**2-na)/(sigma_f**2)
    Wc = ((sigma_f**2-na)/(sigma_f**2)+3-(sigma_f**2/na))
    
    fxw = lambda x,u,Xw: f(x,u,dt) + Xw
    hxv = lambda x,u,Xv: np.hstack((np.sqrt(x[0]**2 + x[2]**2),math.atan2(x[0],x[2]))) + Xv
    
    Wm = np.hstack((Wm,np.ones((2*na))/(2*sigma_f**2)))
    Wc = np.hstack((Wc,np.ones((2*na))/(2*sigma_f**2)))
    dic_state = {}

    for i in index:
        
        X = dic_X[i]
        P = dic_P[i]
        Ifcontain = y.__contains__(i)
        if Ifcontain:
            dic_X[i], dic_P[i]= SPF(X,P,Wm,Wc,u,y[i].reshape(-1,1),sigma_f,nx,nw,nv,fxw,hxv)
        else:
            print ("Mearsurement for", i ,"th people does not exsit")
            
        dic_state[i] =  dic_X[i][0:nx,0]
    
    
    return index,dic_X,dic_P,dic_state


def SPF(X,P,Wm,Wc,u,y,sigma_f,nx,nw,nv,f,h):

#  INPUT
#       X:      n-by-3 state variable n = nx+nw+nv
#       P:      n-by-n state variable P = [[Pxx, 0, 0]]
#                                          [0,   Q, 0]
#                                          [0,   0, R]]
#       Wm:     1-by-2*na+1 weights
#       Wc:     1-by-weights
#       u:      whatever-by-1 input
#       y:      m-by-1 measurement
#       sigma_f:sigma_f value for SPF
#       f:      dynamic model function f(x,u) (+process noise)
#       h:      measurement model function h(x) (+measuremnet noise)

#   OUTPUT
#      X:      n-by-1 state variable n = nx+nw+nv
#      P:      n-by-n state variable P = [[Pxx, 0, 0]]
#                                          [0,   Q, 0]
#                                          [0,   0, R]]


#Seperate vairable
    na = nw+nv+nx
    Xxi = X[:nx,:]
    Xwi = X[nx:(nw+nx),:]
    Xvi = X[(nw+nx):,:]  
    Pxx = P[:nx,:nx]
    Q =  P[nx:(nx+nw),nx:(nx+nw)]
    R = P[(nx+nw):,(nx+nw):]
    
#Predict
    Xxi_ = np.zeros((nx,na*2+1))
    for i in range(2*na+1):  
        #print("Xxi_i:")
        #print(Xxi[:,i])
        Xxi_[:,i] = f(Xxi[:,i],u,Xwi[:,i])
    #print('Xxi_')
    #print(Xxi_)
    xhat_ = np.zeros((nx,1))
    for i in range(2*na+1):
        xhat_ = xhat_+Wm[i]* (np.array([Xxi_[:,i]])).T       #nx x 1  
    #print('xhat_')
    #print(xhat_)
    Pxx_ = np.zeros((nx,nx))
    for i in range(2*na+1):
        Pxx_ = Pxx_+Wc[i]*np.dot( (np.array([Xxi_[:,i]])).T-xhat_, np.array([Xxi_[:,i]])-xhat_.T) 
    #print('Pxx_')
    #print(Pxx_)
#update
    m = y.size;
    yi_ = np.zeros((m,na*2+1))
    for i in range(2*na+1):
        if m == 1:
            print("measurement is theta only")
            yi_[:,i] = h(Xxi[:,i],u,Xvi[:,i])[1]
        else:
            yi_[:,i] = h(Xxi[:,i],u,Xvi[:,i])
            
        
    #print('yi_')
    #print(yi_)
    yhat_ = np.zeros((m,1))
    for i in range(2*na+1):
        yhat_ = yhat_+Wm[i]* (np.array([yi_[:,i]])).T
    #print('yhat_')
    #print(yhat_)
    Pyy_ = np.zeros((m,m))
    for i in range(2*na+1):
        Pyy_ = Pyy_+ Wc[i]*np.dot((np.array([yi_[:,i]])).T-yhat_, np.array([yi_[:,i]])-yhat_.T)
    #print('Pyy_')
    #print(Pyy_)
    Pxy_ = np.zeros((nx,m))
    for i in range(2*na+1):
        Pxy_ = Pxy_+ Wc[i]*np.dot( (np.array([Xxi_[:,i]])).T-xhat_, np.array([yi_[:,i]])-yhat_.T)
    
    #print('Pxy_')
    #print(Pxy_)
    K = np.dot(Pxy_,(np.linalg.inv(Pyy_)))
    #print('K')
    #print(K)
    xhat = xhat_ + np.dot(K,(y-yhat_))
    #print('xhat')
    #print(xhat)
    Pxx = Pxx_ - np.dot(np.dot(K,Pyy_),K.T)
    #print('Pxx')
    #print(Pxx)
    xhata = np.vstack((xhat,np.zeros((nw,1)),np.zeros((nv,1))))
    #print('xhata')
    #print(xhata)
    aaa = np.hstack((Pxx,np.zeros((nx,nw)),np.zeros((nx,nv))))
    bbb = np.hstack((np.zeros((nw,nx)), Q,np.zeros((nw,nv))))
    ccc = np.hstack((np.zeros((nv,nx)), np.zeros((nv,nw)), R))
    P = np.vstack((aaa, bbb, ccc))
    P[np.isnan(P)] = 1E3
    P[P<=0] = 1E-6
    #print('P')
    #print(P)
    ddd = np.dot(xhata,np.ones((1,na))) - sigma_f*sqrtm(P).real
    eee = np.dot(xhata,np.ones((1,na))) + sigma_f*sqrtm(P).real
    #print('a')
    #print(np.dot(xhata,np.ones((1,na))))
    #print('b')
    #print(sqrtm(P))
    X = np.hstack((xhata,ddd,eee))
    #print('X')
    #print(X)
    
    return X,P