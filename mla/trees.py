from collections import Counter, deque
import numpy as np
from mla.metrics import gini_index


# Decision Tree Node
class DT_Node():
    def __init__(self, split_index, split_value):
        # split parameter
        self.split_index = split_index
        self.split_value = split_value
        self.left = None # Class DT_Node
        self.right = None 
        
        # train parameter
        self.leftdata = None # tuple (b_left_x, b_left_y)
        self.rightdata = None
        
        # predict parameter
        self.label = None
        self.halt = False

# Decision Tree
class DecisionTree():
    def __init__(self, maxdepth=float("inf")):
        self.root = None
        self.depth = 0
        self.maxdepth = maxdepth
        
    # Split data into two groups according to the attribute and corresponding value
    def single_split(self, index, val, X, Y):
        '''
        #input
        index: split variable index of X.columns
        val: split value corresponding split variable
        X: numpy array of node data
        Y: numpy array of node label
        #output
        left_x
        left_y
        right_x
        right_y
        '''
        left_flag = X[:,index]<val
        right_flag = np.logical_not(left_flag)
        left_x = X[left_flag]
        left_y = Y[left_flag]
        right_x = X[right_flag]
        right_y = Y[right_flag]
        return (left_x, left_y, right_x, right_y)
    
    # Itrerate all the parameter and all the values to find best split parameter and value
    def get_split(self, X, Y):
        '''
        #input
        X: numpy array
        Y: numpy array
        #output
        DT_Node
        (b_left_x, b_left_y)
        (b_right_x, b_right_y)
        '''
        # halt condition and leaf node
        if len(Counter(Y))==1 or self.depth==self.maxdepth :
            node = DT_Node(None, None)
            node.halt = True
            node.label = Counter(Y).most_common()[0][0]
            return node
        # find best split
        b_split_index = None
        b_split_val = None
        lowest_gini = float("inf")
        for split_index in range(X.shape[1]):
            for split_val in X[:,split_index]:
                # current split
                left_x, left_y, right_x, right_y = self.single_split(split_index, split_val, X, Y)
                # curr gini
                gini = len(left_y)*gini_index(left_y)+len(right_y)*gini_index(right_y)
                #print('X%d < %.3f Gini=%.3f' % ((split_index), split_val, gini))
                if gini<lowest_gini:
                    lowest_gini = gini
                    b_split_index = split_index
                    b_split_val = split_val
                    b_left_x, b_left_y, b_right_x, b_right_y = left_x, left_y, right_x, right_y
        #print('Best Split: [X%d < %.3f]' % (b_split_index, b_split_val) )
        node = DT_Node(b_split_index, b_split_val)
        node.leftdata = (b_left_x, b_left_y)
        node.rightdata = (b_right_x, b_right_y)    
        return node
    
    # Traing-----------------------------------
    def train(self, X, Y):
        self.root = self.get_split(X, Y)
        queues = deque([self.root])
        while queues:
            self.depth += 1
            for i in range(len(queues)):
                node = queues.popleft()
                if node.halt:
                    continue
                node.left = self.get_split(node.leftdata[0], node.leftdata[1])
                node.right = self.get_split(node.rightdata[0], node.rightdata[1])
                queues.append(node.left)
                queues.append(node.right)
        
    # Prediction-----------------------------------
    def predict(self, X):
        rst = []
        for row in X:
            node = self.root
            while True:
                # halt condition
                if node.halt:
                    rst.append(node.label)
                    break
                # recuresive    
                split_index = node.split_index
                split_value = node.split_value
                if row[split_index]<split_value:
                    node = node.left
                else:
                    node = node.right
        return rst