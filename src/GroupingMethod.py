# 0. import the needed packages
import numpy as np
import operator
from scipy.stats.distributions import chi2
def rth2xy(x):
    r = x[0]
    th = x[1]
    return np.array([r*np.cos(th),r*np.sin(th)])



def deleteTracker(tracker,tracker_Feature,tracker_Feature_P,i):
    
    del tracker_Feature[tracker[i]]
    del tracker_Feature_P[tracker[i]]
    del tracker[i]

'''
def getClassifier(measures,features_measure,tracker,tracker_Feature,tracker_Feature_P,threshold):
    #measures: debug only!
    detected_class = -2*np.ones(len(features_measure))
    detected_class_dist = -2*np.ones(len(features_measure))
    for measureIndex,feature in enumerate(features_measure):
        #feature = getFeatureFromMeasure(measure,u)
        classifier = -2         
        dis = -2
        if (feature.size != 0):
            classifier,dis = PNN(feature,tracker,tracker_Feature,tracker_Feature_P,threshold)
            # classifier = -1 for unclassified measurement    
                
            #update detected_class that stores the overall detected results for this time step
        print("new measure:")
        print(measures[measureIndex][0])
        print("with feature:")
        print(feature)
        print("is classified to:")
        print(classifier)
        print("with feature:")
        if classifier != -1:
            print(tracker_Feature[classifier])
        else:
            print("you pick up one")
        print("All features:")
        print(tracker_Feature)
        print()
        if (len(features_measure) != 0):
            detected_class[measureIndex] = classifier
            detected_class_dist[measureIndex] = dis
                    
            
    return detected_class
'''

def getClassifier2(features_measure,tracker,tracker_Feature,tracker_Feature_P,thresholdPercent):
    detected_class = -1*np.ones(len(features_measure))
    detected_class_dist = -1*np.ones(len(features_measure))
    dicts_dist = {}
    dicts_dist_sorted = {}
    #run PNN to calculate dictionary of distance
    for measureIndex,feature in enumerate(features_measure):
        if (feature.size != 0):
            #Check Actual Measure Feature Size!!!!!
            if feature.size == 1:
                #if measurement is theta only, change the size of feature and corresponding feature P
                threshold = chi2.ppf(thresholdPercent, df=1)*30
                specialTracker_Feature = {}
                specialTracker_Feature_P = {}
                for candidate in tracker:
                    specialTracker_Feature[candidate] = tracker_Feature[candidate][1]
                    specialTracker_Feature_P[candidate] = np.array([tracker_Feature_P[candidate][1,1]*3])
                dicts_dist[measureIndex] = PNN_dict(feature,tracker,specialTracker_Feature,specialTracker_Feature_P,threshold)
            else:
                threshold = chi2.ppf(thresholdPercent, df=2)*10
                dicts_dist[measureIndex] = PNN_dict(feature,tracker,tracker_Feature,tracker_Feature_P,threshold)
                
            dicts_dist_sorted[measureIndex] = sorted(dicts_dist[measureIndex],key = dicts_dist[measureIndex].get)
    '''
    print("dicts of dist:")
    print(dicts_dist)
    print("sorted class order:")
    print(dicts_dist_sorted)
    '''
    #deal with duplication and get result
    isDuplicated = True
    while isDuplicated:
        isDuplicated = False
        for  measureIndex,feature in enumerate(features_measure):
            # for each result corresponding to measurement to be returned:
            '''
            print()
            print("sorted class order for measure:")
            print(measureIndex)
            print(dicts_dist_sorted[measureIndex])
            '''
            if len(dicts_dist_sorted[measureIndex]) != 0:
                #if distance dictionary for that measurement is not empty:
                detected_class[measureIndex] = dicts_dist_sorted[measureIndex][0]
           
                if dicts_dist[measureIndex][detected_class[measureIndex]] > threshold:
                    #if current best distance does not satisfy threshold, result is -1
                    detected_class[measureIndex] = -1
                #loop detected class to check duplication
                for detected_class_index in range(len(detected_class)):
                    #compare current measurement distance array to all previous measurements array to check for duplication
                    if (measureIndex > detected_class_index 
                        and detected_class[detected_class_index] == detected_class[measureIndex] 
                        and detected_class[measureIndex] != -1):
                        '''
                        print("duplicated measure:")
                        print(detected_class_index)
                        
                        print("at class:")
                        print(detected_class[detected_class_index])
                        print("dist:")
                        '''
                        dist1 = dicts_dist[detected_class_index][detected_class[detected_class_index]]
                        dist2 = dicts_dist[measureIndex][detected_class[measureIndex]]
                        '''
                        print(dicts_dist[detected_class_index][detected_class[detected_class_index]])
                        print("and my dist:")
                        print(dicts_dist[measureIndex][detected_class[measureIndex]])
                        print("dist_sorted:")
                        print(dicts_dist_sorted[detected_class_index])    
                        print("my dist_sorted:")
                        print(dicts_dist_sorted[measureIndex])  
                        '''
                        if dist1 > dist2:
                            #print("to be deleted:")
                            #print(dicts_dist_sorted[detected_class_index][0])
                            if len(dicts_dist_sorted[detected_class_index]) > 0:
                                del dicts_dist_sorted[detected_class_index][0]
                            else:
                                detected_class[detected_class_index] = -1
                                break
                            '''
                            print("so deteled dist_sorted:")
                            print(dicts_dist_sorted[detected_class_index])
                            '''
                        else:
                            #print("to be deleted:")
                            #print(dicts_dist_sorted[measureIndex][0])
                            if len(dicts_dist_sorted[measureIndex]) > 0:
                                del dicts_dist_sorted[measureIndex][0]
                            else:
                                detected_class[measureIndex] = -1
                                break
                            '''
                            print("so deteled my dist_sorted:")
                            print(dicts_dist_sorted[measureIndex])
                            '''

                        isDuplicated = True    
                    
                    
            
            else:
                detected_class[measureIndex] = -1
        
    
                    
            
    return detected_class




'''
def PNN(x,classes,features,feature_P_matrix,threshold=1):
    # return the class of detect and the score
    

    num_of_classes = len(classes)
    dictionary_of_distance = {}

    #compare x with candidates in classes
    if (len(classes)!= 0):
        for k in range(0,num_of_classes):
            temp_summnation = 0.0
            feature = features[classes[k]]
            m = feature
            #Implementation of getting Gaussians 
            if m != x:
                #Do something
            x_m = x-m
            C = feature_P_matrix[classes[k]]
            dictionary_of_distance[k] = -1*np.dot(x_m.transpose(),np.dot(np.linalg.inv(C),x_m))
             
    #Get the classified class
    distance = -1
    #if had something to compare with
    if(len(dictionary_of_distance) != 0):
        #find the best compare
        classified_class_index = max(dictionary_of_distance, key=dictionary_of_distance.get)
        #check if the best compare is rejected or not
        
        if(dictionary_of_distance[classified_class_index] < -1*threshold):
            #if the best compare is rejected, it is a new detection
            
            print("best compare rejected with distance:")
            print(dictionary_of_distance[classified_class_index])
            print("threshold:")
            print(threshold)
            print("feature:")
            print(features[classes[classified_class_index]])
            print("C:")
            print(feature_P_matrix[classes[classified_class_index]])
            print("x_m:")
            print(x-features[classes[classified_class_index]])
            print("C^-1:")
            
            C = feature_P_matrix[classes[classified_class_index]]
            print(np.linalg.inv(C))
            x_m = x-features[classes[classified_class_index]]
            print("dist:")
            print(np.dot(x_m.transpose(),np.dot(np.linalg.inv(C),x_m)))
            
            
            classified_class = -1
        else:
            #if the best compare is valid, get id, and manipulate trackers & untracked            
            classified_class = classes[classified_class_index]
            distance=-1*dictionary_of_distance[classified_class_index]
    #if had nothing to compare with, its a new detection
    else:
        classified_class = -1
    return classified_class,distance
'''
# ---- END OF THE CODE ------

def PNN_dict(x,classes,features,feature_P_matrix,threshold=1):
    # return the class of detect and the score
    

    num_of_classes = len(classes)
    dictionary_of_distance = {}

    #compare x with candidates in classes
    if (len(classes)!= 0):
        for k in range(0,num_of_classes):
            temp_summnation = 0.0
            feature = features[classes[k]]
            m = feature
            #Implementation of getting Gaussians 
            x_m = x-m
            C = feature_P_matrix[classes[k]]
            if C.shape[0]>1:
                dictionary_of_distance[classes[k]] = np.dot(x_m.transpose(),np.dot(np.linalg.inv(C),x_m))
            else:
                dictionary_of_distance[classes[k]] = (x_m**2)*C
             

    return dictionary_of_distance



