import numpy as np
import GroupingMethod as gp
import SPF


def f(X,u,dt):
    #Predit the relative postion from ego to target at t[k+1] in x,y coordinate
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
    return np.array([X_k[0],X_k[2]])



def getFeatureFromState(X,u,dt):
    #transfer feature from x,y coord to r,th coord
    PosXY = f(X,u,dt)
    x = PosXY[0]
    y = PosXY[1]
    
    return np.array([(x**2+y**2)**0.5,np.arctan(y/x)])




def getFeatureFromMeasure(x,u):
    #x: measurement [r,th,...]
    x =x[0]
    if x.size==0: return np.array([])   
    elif x.size == 1: return np.array([x[0]])   
    else:
        r = x[0]
        th = x[1]
        return np.array([r,th])   



def initialXFromMeasure(measure,u):
    z = measure[0]
    if len(z) == 2:
        r = z[0]
        th = z[1]
        
    else:
        r = measure[1]
        th = z[0]
    return np.array([r*np.sin(th),0,r*np.cos(th),0])


def getZFromMeasure(measure):
    return measure[:][0]
    
    
    
def initialFeatureFromMeasure(measure,u):
    #if r at measure[0] is missing, replace it with measure[1] so that if lidar detection is Nan, use range from camera to initialize feature
    x =measure[0]
    if x.size==0: return np.array([])   
    elif x.size == 1: return np.array([x[0],measure[1]])   
    else:
        r = x[0]
        th = x[1]
        return np.array([r,th])   
    

def findnewlabel(tracker):
    if len(tracker) == 0:
        return 1
    else:
        return max(tracker)+1

    
class tracker:
    # tracker features: [r,th]


    def __init__(self):
        #tracked item properties
        self.mytracker = []
        self.mytrackerSeen = {}
        self.mytracker_Feature = {}
        self.mytracker_Feature_P = {}
        self.mytracker_X = {}
        self.mytracker_P = {}
        self.mytracker_state = {}
        
        
        
        
    def getTrackerState(self):
        return [self.mytracker, self.mytracker_state]
    
    def getEffectiveTrackerState(self):
        effectiveTracker = [tracker for tracker in self.mytracker if self.mytrackerSeen[tracker] >= 3]
        effectiveTrackerState = {}
        for tracker in effectiveTracker:
            effectiveTrackerState[tracker] = self.mytracker_state[tracker]
        
        return [effectiveTracker,effectiveTrackerState]
       
        
    def updateTracker_Feature(self,tracker,tracker_state,tracker_P,u,dt):
        for trackerIndex in tracker:
            self.mytracker_Feature[trackerIndex][0:2] = getFeatureFromState(tracker_state[trackerIndex].flatten()[0:4],u,dt)
            #P1 = tracker_P[trackerIndex][0:4:2,0:4:2];
            #P1 = P1/np.sum(P1)*100
            P1 = np.array([[1,0],[0,1]])*25
            P2 = np.eye(self.mytracker_Feature[trackerIndex].size-2)*3
            aaa = np.hstack((P1,np.zeros((P1.shape[0],P2.shape[1]))))
            bbb = np.hstack((np.zeros((P2.shape[0],P1.shape[1])),P2))
            self.mytracker_Feature_P[trackerIndex] = np.vstack((aaa,bbb))
        
    def updateTracker(self,effective_Z,u,dt = 1/30):
        #not finished
        tracker = self.mytracker
        tracker_X = self.mytracker_X
        tracker_P = self.mytracker_P
        sigma_f = 1.5
        index,dic_X,dic_P,self.mytracker_state = SPF.multiPeopleSPF(tracker,tracker_X,tracker_P,u,effective_Z,sigma_f,dt)
        self.mytracker = tracker
        self.mytracker_X = dic_X
        self.mytracker_P = dic_P
        self.updateTracker_Feature(tracker,self.mytracker_state,self.mytracker_P,u,dt)
                             
    def initializeTracker(self,candidate,z0,measure,u,dt = 0.2):
        sigma_wx, sigma_wy = 0.05, 0.05
        sigma_vr, sigma_vth = 0.05, 0.01
        sigma_wx_ego, sigma_wy_ego = 0.0001, 0.0001
        sigma_vx_ego, sigma_vy_ego = 0, 0
        sigma_f = 1.5
        nx = 4
        nw = 4
        nv = 2
        na = nx+nw+nv 
        Pxx0 = np.eye(4)*2
        varX = sigma_wx*sigma_wx+sigma_wx_ego*sigma_wx_ego
        varY = sigma_wy*sigma_wy+sigma_wy_ego*sigma_wy_ego
        varVx = 0.1
        varVy = 0.1
        varX = varX * dt
        varY = varY * dt
        x0 = initialXFromMeasure(measure,u)
        x0 = np.array([x0])
        Q = np.array([[varX,dt*varX,0,0], [dt*varX,varVx,0,0], [0,0,varY,dt*varY], [0,0,dt*varY,varVy]])
        R = np.array([[sigma_vr*sigma_vr,0],[0,sigma_vth*sigma_vth]])     
        aaa = np.hstack((Pxx0, np.zeros((nx,nw)), np.zeros((nx,nv))))
        bbb = np.hstack((np.zeros((nw,nx)), Q, np.zeros((nw,nv))))
        ccc = np.hstack((np.zeros((nv,nx)), np.zeros((nv,nw)), R))
        P = np.vstack((aaa,bbb,ccc))                         
        xhata0 = (np.hstack((x0, np.zeros((1,nw)), np.zeros((1,nv))))).T
        X = np.hstack(
            (xhata0, np.dot(
                xhata0,np.ones((1,na)))-sigma_f*np.sqrt(P), np.dot(xhata0,np.ones((1,na)))+sigma_f*np.sqrt(P)))
        
        #Uptade tracker properties
        self.mytrackerSeen[candidate] = 3
        self.mytracker_X[candidate] = X
        self.mytracker_P[candidate] = P
        self.mytracker_Feature[candidate] = initialFeatureFromMeasure(measure,u)
        self.mytracker_Feature_P[candidate] = np.eye(self.mytracker_Feature[candidate].size)*10
    
        
        
    def deleteTracker(self,candidate):
        del self.mytrackerSeen[candidate]
        del self.mytracker_Feature[candidate]
        del self.mytracker_Feature_P[candidate]
        del self.mytracker_X[candidate]
        del self.mytracker_P[candidate]
        
    
    def track(self,measures,u,dt):
       
        # measures: list [num of detected obj x (nparray(r, th) or nparray(th),r_camera, ...) ]
        # u: car status input [V,w]
   
        number_of_features  = 2
        tracker = self.mytracker
        tracker_Feature_P = self.mytracker_Feature_P
        tracker_Feature = self.mytracker_Feature
        tracker_seen = self.mytrackerSeen
        
        measure_size = len(measures)
        
        detected_class = -1*np.ones(measure_size) #store detected classes represented as each classifier
        thresholdPercent = 0.95
        effective_Z = {}
        Zs = [];
        features_measure = [];
        for measure in measures:
            Zs.append(getZFromMeasure(measure))
            features_measure.append(getFeatureFromMeasure(measure,u))
            
        
            
        detected_class = gp.getClassifier2(features_measure,tracker,tracker_Feature,tracker_Feature_P,thresholdPercent)
        
        
        #update status for tracker
        for candidate in tracker:
            isseen = 0
            for meausure_index,detected in enumerate(detected_class):
                if candidate == detected:
                    isseen = 1
                    effective_Z[candidate] = Zs[meausure_index]
                    if tracker_seen[candidate] < 3:
                        tracker_seen[candidate] = tracker_seen[candidate]+1
                    break
            if isseen == 0:
                tracker_seen[candidate] = tracker_seen[candidate] - 1
                
        #delete trackers candidades that are not seen for a while
  
        tracker_size = len(tracker)
        i = 0
        while i < tracker_size:
            candidate = tracker[i]             
            if tracker_seen[candidate] <0:
                self.deleteTracker(candidate)
                del tracker[i]
                i = i - 1
                tracker_size = tracker_size - 1   
            i = i + 1        

        #deal with new detections:
        for meausure_index,detected in enumerate(detected_class):
            if detected == -1:
                label = findnewlabel(tracker)
                tracker.append(label)
                tracker_seen[label] = 1
                self.initializeTracker(label,Zs[meausure_index],measures[meausure_index],u,dt)
                
        self.updateTracker(effective_Z,u,dt)

            

        
        return detected_class
        
