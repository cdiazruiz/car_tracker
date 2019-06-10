from detector import ROSdetector
import tracker
import SPF
import numpy as np
import matplotlib.pyplot as plt
#import skimage.io 
import cv2





def projectLidar(lidar, Tr):
    lidar = np.hstack((np.array(lidar),np.ones((len(lidar),1))))
    lidar = np.dot(Tr,lidar.T)
    return lidar.T







class PedestrainTracker:
    def __init__(self):
        self.myTracker = tracker.tracker()
        self.pedestrains = {}
        
    def track(self,projection,detections,segResults,u,image,lidar,dt):
        myTracker = self.myTracker
        pedestrains = self.pedestrains
        measures = ROSdetector.getDetection(projection,detections,segResults)
        #u = [0,0]
        classification = myTracker.track(measures[1],u,dt)
        print('classification:')
        print(classification)
        print("Track result:")
        print(myTracker.getEffectiveTrackerState())
        
        
        
        pedestrainIndexs, pedestrainStates = myTracker.getEffectiveTrackerState()
        for pedestrainIndex in pedestrainIndexs:
            if pedestrains.__contains__(pedestrainIndex):
                pedestrains[pedestrainIndex] = np.vstack((pedestrains[pedestrainIndex],pedestrainStates[pedestrainIndex]))
            else:
                pedestrains[pedestrainIndex] = pedestrainStates[pedestrainIndex]
        
        #img
        boxes = measures[2]
        #To Be Changed
        #image = skimage.io.imread(os.path.join(IMAGE_DIR, imageName))    
        #End To Be Changed
        boxXPos = []
        for box in boxes:
            boxXPos.append(box[1])
        boxXPos = np.array(boxXPos)
        boxOrder = np.argsort(boxXPos)
    
            
            
        for measureIndex,box in enumerate(boxes):
            textHeight = 140*(np.where(boxOrder == measureIndex)[0]%2)
            cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),(255,0,0),3)
            text = "ID: "+str(classification[measureIndex])
            cv2.putText(image,text,(box[1],box[0]-60-textHeight), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), lineType=cv2.LINE_AA)
            #print(measures[1][2][0])
            if measures[1][measureIndex][0].size > 1:
                r = measures[1][measureIndex][0][0]
                th = measures[1][measureIndex][0][1]
                text = "measured state: " + str(np.array([r*np.sin(th),0,r*np.cos(th),0]).round(2))
            else:
                th = measures[1][measureIndex][0][0]
                text = "measured state (theta only): " + str(th)
            cv2.putText(image,text,(box[1],box[0]-30-textHeight), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), lineType=cv2.LINE_AA)
            if pedestrainIndexs.__contains__(classification[measureIndex]):
                text = "filterd state: " + str(pedestrainStates[classification[measureIndex]].round(2))
                cv2.putText(image,text,(box[1],box[0]-textHeight), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), lineType=cv2.LINE_AA)
        #To Be Changed    
        #skimage.io.imsave(os.path.join(IMAGE_OUT_DIR, imageName),image)
        #End To Be Changed
        Tr = projection.transformation_matrix
        projectedLidar = projectLidar(lidar, Tr)
        projectedLidar = np.array(projectedLidar)
        
        print("====================== Tracking Finished =============================")
        return [projectedLidar,image,pedestrainIndexs, pedestrainStates,pedestrains]
        
        
