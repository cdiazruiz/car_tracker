class segResults:
    def __init__(self,instanceIDs,segResultPts):
        self.instanceIDs = instanceIDs
        self.segResultPts = segResultPts
        
        
    def getPtCLoudsFromInstanceId(self,instanceID):
        if instanceID in self.instanceIDs:
            Index = self.instanceIDs.index(instanceID)
            return self.segResultPts[Index]
        else:
            return []
        
    def getPtCLoudsFromInstanceIds(self,instanceIDs):
        returnResult = []
        for instanceID in instanceIDs:
            returnResult.append(self.getPtCLoudsFromInstanceId(instanceID))   
        return returnResult

    def getAllPtClouds(self):
        return self.segResultPts
