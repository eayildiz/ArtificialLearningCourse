from typing import List

#TODO: Create a new class for Node
class Node:
    def __init__(self):
        pass

class DecisionTreeClassifier:
    def __init__(self, max_depth: int):
        #TODO: Create a root for tree.
        pass

    #region Functions for fit function.
    #self, X: List[List[float]], y: List[int]
    def CheckInputFit(self, X: List[List[float]], y: List[int]):
        if(len(X) <= 0 or len(y) <= 0):
            return True
        return False

    def NumberOfSpeciesInTheNode(self, y: List[int]):
        numberOf0 = 0
        numberOf1 = 0
        numberOf2 = 0
        for species in y:
            match species:
                case 0:
                    numberOf0 = numberOf0 + 1
                case 1:
                    numberOf1 = numberOf1 + 1
                case 2:
                    numberOf2 = numberOf2 + 1
        return [numberOf0, numberOf1, numberOf2]

    def GiniImpurityCalculationForTheNode(self, y: List[int], curretnCountInstance: int):
        numberOfSpecies = self.NumberOfSpeciesInTheNode(self, y)
        Species0Probability = numberOfSpecies[0] / curretnCountInstance
        Species1Probability = numberOfSpecies[1] / curretnCountInstance
        Species2Probability = numberOfSpecies[2] / curretnCountInstance
        return 1 - (Species0Probability**2 + Species1Probability**2 + Species2Probability**2)

    def splitCurrentNode(self, feature: int, rowIndexOfInstance: int, X: List[List[float]], y: List[int]):
        left_train = []
        left_target = []
        right_train = []
        right_target = []
        keyValue = X[rowIndexOfInstance][feature]
        for row in X:
            if(X[row][feature] <= keyValue):
                left_train.insert(X[row])
                left_target.insert(y[row])
            else:
                right_train.insert(X[row])
                right_target.insert(y[row])
        return left_train, left_target, right_train, right_target
    
    def findOptimalSplit(self, X: List[List[float]], y: List[int], curretnCountInstance: int):
        minWeightedGini = 1
        feauterOfMin = -1
        rowOfMin = -1
        for feature in range(0,4):
            for row in range(len(y)):
                left_train, left_target, right_train, right_target = self.splitCurrentNode(self, feature, row, X, y)
                #If any split won't happen, then that split has no meaning, need to continue to search.
                if(self.CheckInputFit(self, left_target, left_train) or self.CheckInputFit(self, right_target, right_train)):
                    continue
                leftGini = self.GiniImpurityCalculationForTheNode(self, left_target, len(left_target))
                rightGini = self.GiniImpurityCalculationForTheNode(self, right_target, len(right_target))
                weightedGini = leftGini * (len(left_target) / curretnCountInstance) + rightGini * (len(right_target) / curretnCountInstance)
                if(weightedGini < minWeightedGini):
                    feauterOfMin = feature
                    rowOfMin = row
                    minWeightedGini = weightedGini
        return feauterOfMin, rowOfMin
    #endregion

    def fit(self, X: List[List[float]], y: List[int]):
        if(self.CheckInputFit(self, X, y)):
            print("Wrong Input!")
            return
        curretnCountInstance = len(y)
        initialGiniImpurity = self.GiniImpurityCalculationForTheNode(self, y, curretnCountInstance)
        
        if(initialGiniImpurity == 0):
            return
        #Continue to create new nodes for decision tree.
        feauterOfMin, rowOfMin = self.findOptimalSplit(self, X, y, curretnCountInstance)
        #TODO: Save this node's properties. HERE!
        left_train, left_target, right_train, right_target = self.splitCurrentNode(feauterOfMin, rowOfMin, X, y)
        self.fit(left_train, left_target)
        self.fit(right_train, right_target)
        



    def predict(self, X: List[List[float]]):
        pass





# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)   





# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    