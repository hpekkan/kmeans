from typing import List


def unique(arr):
   unique_set = set(arr)
   return (list(unique_set))
def deep_copy2d(copy):
    return [[i for i in row] for row in copy]
def deep_copy1d(copy):
    return [i for i in copy]

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,impurity=None,val=None):
        self.feature=feature
        self.threshold=threshold
        self.left = left
        self.right = right
        self.impurity=impurity
        self.val = val
    def __str__(self):
        if self.val is not None:
            return str(self.val)
    



class DecisionTreeClassifier:

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: List[List[float]], y: List[int]):
        if(len(X)<1): return;
        r=len(X)
        c=len(X[0])
        data = deep_copy2d(X)
        for i in range(len(y)):
            data[i].append(y[i]) 
        self.tree = self.initializeTree(data,r,c,0)
    
    def initializeTree(self,data,r,c,depth=0):
        y=[Y[c] for Y in data]
        r = len(y)
        if(r>1 and depth<=self.max_depth):
            split = self.find_split(data,c)
            if split["impurity"]>0:
                left = self.initializeTree(split["left_dataset"],r,c,depth+1)
                right = self.initializeTree(split["right_dataset"],r,c,depth+1)
                return Node(split["feature"],split["threshold"],left,right,split["impurity"])
        leaf_class = self.find_classes(y)
        return Node(val=leaf_class)
       
        
    def find_split(self,data,c):
        split = {}
        max_impurity = -float("inf")
        
        for column in range(c):
            column_values = [values[column] for values in data]
            possible_thresholds = unique(column_values)
            for threshold in possible_thresholds:
                left_d,right_d=self.split(data,column,threshold)
                if left_d is not None and right_d is not None:
                    y = [row[c] for row in data]
                    left_y = [row[c] for row in left_d]
                    right_y = [row[c] for row in right_d]
                    
                    curr_impurity = self.impurity(y,left_y,right_y,c)
                    if curr_impurity > max_impurity:
                        split["feature"]=column
                        split["threshold"]=threshold
                        split["left_dataset"]=left_d
                        split["right_dataset"]=right_d
                        split["impurity"]=curr_impurity
                        max_impurity = curr_impurity
        return split
                    
            
    def split(self,X,column,threshold):
        X_l = [cursor for cursor in X if cursor[column]<=threshold]
        X_r = [cursor for cursor in X if cursor[column]>threshold]
        return X_l,X_r
    
    def impurity(self,parent_d,left_d,right_d,c):
        left_weight = len(left_d) / len(parent_d)
        right_weight = len(right_d)/ len(parent_d)
        return self.gini_impurity(parent_d,c) - (left_weight*self.gini_impurity(left_d,c) + right_weight*self.gini_impurity(right_d,c))

    def gini_impurity(self,y,c):
        classes = unique(y)
        gini = 0
        for cur_cls in classes:
            curr_count = 0
            for y_cls in y:
                if y_cls == cur_cls: curr_count+=1
            prob_cls = curr_count/len(y)
            gini += prob_cls**2
        return 1-gini    
    def find_classes(self,y):
        count = {}
        for i in range(len(y)):
            if y[i] not in count:
                count[y[i]]=1
            else:
                count[y[i]]+=1
        answer,local_max = -1,-1
        for cur in count:
            if count[cur]>local_max:
                local_max = count[cur]
                answer = cur
        return answer
    
    
    def printTree(self):
        self.printTreeAux(self.tree,0)
    def printTreeAux(self,root,k):
        print("."*k,end="")
        if root.val is not None:
            print("Decided Class:",root.val)
                
        else:
            feature = "SepalLengthCm" if root.feature == 0 else "SepalWidthCm" if root.feature == 1 else "PetalLengthCm" if root.feature == 2  else "PetalWidthCm" 
            print("Column:",feature,"threshold <=",root.threshold, "impurity:",root.impurity)
            self.printTreeAux(root.left,k+1)
            self.printTreeAux(root.right,k+1)
    
    def predict(self, X: List[List[float]]):
        predictions = [self.predictAux(x, self.tree) for x in X]
        return predictions
    def predictAux(self, X, root):
        if root.val!=None: return root.val
        col = X[root.feature]
        if col<=root.threshold:
            return self.predictAux(X, root.left)
        else:
            return self.predictAux(X, root.right)


if __name__ == '__main__':  
    print()
    #X, y = ...
    #X_train, X_test, y_train, y_test = ...
    #clf = DecisionTreeClassifier(max_depth=5)
    #clf.fit(X_train, y_train)
    #yhat = clf.predict(X_test)    