from typing import List

# If the answer is yes, go to left; else go right.
# If you reach a leaf then it means prediction is over.
class Node:
    maxDepth = -1
    def __init__(self, depthOfNode: int = 0, parent: "Node" = None):
        self.featureOfSelection = -1        # This is index of column of original train set. ID is dropped.
        self.valueOfSelection = -1.0        # This is value of seperator. We will compare with this when predicting.
        self.giniValue = 1                  # This is gini impurity of the node. If it is 0, then it is leaf node.
        self.countOfSpecies = [-1, -1, -1]
        self.classOfSpecies = -1
        self.isLeaf = False
        self.depthOfNode = depthOfNode
        self.parent = parent
        self.leftChild = None
        self.rightChild = None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int):
        self.root = Node()
        self.root.maxDepth = max_depth
        self.currentNode = self.root

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
        numberOfSpecies = self.NumberOfSpeciesInTheNode(y)
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
        for rowIndex in range(0, len(X)):
            if(X[rowIndex][feature] <= keyValue):
                left_train.append(X[rowIndex])
                left_target.append(y[rowIndex])
            else:
                right_train.append(X[rowIndex])
                right_target.append(y[rowIndex])
        return left_train, left_target, right_train, right_target
    
    def findOptimalSplit(self, X: List[List[float]], y: List[int], curretnCountInstance: int):
        minWeightedGini = 1
        feauterOfMin = -1
        rowOfMin = -1
        for feature in range(0,4):
            for row in range(len(y)):
                left_train, left_target, right_train, right_target = self.splitCurrentNode(feature, row, X, y)
                # If any split won't happen, then that split has no meaning, need to continue to search.
                if(self.CheckInputFit(left_target, left_train) or self.CheckInputFit(right_target, right_train)):
                    continue
                leftGini = self.GiniImpurityCalculationForTheNode(left_target, len(left_target))
                rightGini = self.GiniImpurityCalculationForTheNode(right_target, len(right_target))
                weightedGini = leftGini * (len(left_target) / curretnCountInstance) + rightGini * (len(right_target) / curretnCountInstance)
                if(weightedGini < minWeightedGini):
                    feauterOfMin = feature
                    rowOfMin = row
                    minWeightedGini = weightedGini
        return feauterOfMin, rowOfMin

    def saveNode(self, node: Node, featureOfSelection: int, valueOfSelection: float, giniValue: int, y: List[int], isLeaf: bool):
        node.featureOfSelection = featureOfSelection
        node.valueOfSelection = valueOfSelection
        node.giniValue = giniValue
        node.countOfSpecies = self.NumberOfSpeciesInTheNode(y)
        node.isLeaf = isLeaf
        if(node.countOfSpecies[0] >= node.countOfSpecies[1] and node.countOfSpecies[0] >= node.countOfSpecies[2]):
            node.classOfSpecies = 0
        elif(node.countOfSpecies[1] >= node.countOfSpecies[0] and node.countOfSpecies[1] >= node.countOfSpecies[2]):
            node.classOfSpecies = 1
        elif(node.countOfSpecies[2] >= node.countOfSpecies[0] and node.countOfSpecies[2] >= node.countOfSpecies[1]):
            node.classOfSpecies = 2
        else:
            node.classOfSpecies = 2
    #endregion

    def fit(self, X: List[List[float]], y: List[int]):
        if(self.CheckInputFit(X, y)):
            print("Wrong Input!")
            return
        curretnCountInstance = len(y)
        initialGiniImpurity = self.GiniImpurityCalculationForTheNode(y, curretnCountInstance)
        
        if(initialGiniImpurity == 0):
            self.saveNode(self.currentNode, -1, -1, initialGiniImpurity, y, True)
            return
        # Continue to create new nodes for decision tree.
        feauterOfMin, rowOfMin = self.findOptimalSplit(X, y, curretnCountInstance)
        #Saving the findings.
        if(self.currentNode.depthOfNode == self.currentNode.maxDepth):
            self.saveNode(self.currentNode, feauterOfMin, X[rowOfMin][feauterOfMin], initialGiniImpurity, y, True)
            return
        self.saveNode(self.currentNode, feauterOfMin, X[rowOfMin][feauterOfMin], initialGiniImpurity, y, False)
        left_train, left_target, right_train, right_target = self.splitCurrentNode(feauterOfMin, rowOfMin, X, y)

        temp_node = Node(depthOfNode=self.currentNode.depthOfNode + 1, parent=self.currentNode)
        self.currentNode.leftChild = temp_node
        self.currentNode = self.currentNode.leftChild
        self.fit(left_train, left_target)

        self.currentNode = self.currentNode.parent  #To go to right child.
        temp_node = Node(depthOfNode=self.currentNode.depthOfNode + 1, parent=self.currentNode)
        self.currentNode.rightChild = temp_node
        self.currentNode = self.currentNode.rightChild
        self.fit(right_train, right_target)
        self.currentNode = self.currentNode.parent  #To go upper for continue to iterating.



    def predict(self, X: List[List[float]]):
        result = []
        for row in X:
            self.currentNode = self.root
            while(self.currentNode.isLeaf == False):
                if(row[self.currentNode.featureOfSelection] <= self.currentNode.valueOfSelection):
                    self.currentNode = self.currentNode.leftChild
                else:
                    self.currentNode = self.currentNode.rightChild
            result.append(self.currentNode.classOfSpecies)
        return result




# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    