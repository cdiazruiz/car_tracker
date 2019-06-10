import numpy as np
import random
import time
import measurement
from operator import itemgetter 

    
class ROSdetector:
    @staticmethod
    def getDetection(projection,detections,segResults):
        shape = detections.shape
        Tr = projection.transformation_matrix
        Proj = projection.projection_matrix
        peopleList = [index for index,value in enumerate(detections.class_names) if value == 'person']
        print("peopleList:")
        print(peopleList)
        if len(peopleList) == 0:
            t = time.time()
            return [t,[],[]]
        #boxes = detections.rois[peopleList]
        #masks = detections.masks[peopleList]
        #segResults = segResults[peopleList]
        print("detected box")
        print(detections.rois)
        print("size of detected box")
        print(len(detections.rois))
        print("size of detected masks")
        print(len(detections.masks))
        print("size of detected lidars")
        print(len(segResults.getAllPtClouds()))
        boxes = list(itemgetter(*peopleList)(detections.rois)) 
        masks = list(itemgetter(*peopleList)(detections.masks)) 
        segResults = segResults.getPtCLoudsFromInstanceIds(peopleList)
        
        if len(peopleList) == 1:
            boxes = [boxes]
            masks =[masks]
            segResults = segResults
            print(boxes)
            print(segResults)
            
        print("detected pedestrain box")
        print(boxes)
        print("size of detected pedestrain masks")
        print(len(masks))
        print("size of detected pedestrain lidars")
        print(len(segResults))
        
        threshold = 20
        returnBox= []
        measures = []
        for i,box in enumerate(boxes):
            #measures.append(measurement.measure(image,box,masks[i]))
            oneMeasure = measurement.measure2(shape,box,masks[i],segResults[i],Tr,Proj)
            
            
            if ((oneMeasure[0].size == 1 and oneMeasure[1] < threshold) 
                or (oneMeasure[0].size == 2 and oneMeasure[0][1] < threshold)):
            # if measurement is in distance:
                measures.append(oneMeasure)
                returnBox.append(box)
        
        t = time.time()
        returnDetection = [t,measures,returnBox]
        return returnDetection
    
            
