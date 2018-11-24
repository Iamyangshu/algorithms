#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
class PLA(object):
  
    #初始化学习率及迭代次数
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    #初始化权重并添加增广向量,添加w_[0]
    def _initialize_w(self,m):
        self.w_=np.zeros(1+m)

    #计算PLA输出
    def net_input(self, X):
        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    #计算依靠PLA输出的分类结果
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

        
    
    #修正权值系数
    def fit(self, X, y):

        self._initialize_w(X.shape[1])
        
        self.errors_ = []#分类误差

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = (target - self.predict(xi))
                self.w_[1:] += self.eta*update * xi
                self.w_[0] += self.eta*update
                errors+=update
            self.errors_.append(errors)
        return self

    

    
    
        
            
            
        
        
        


# In[10]:


import pandas as pd
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
dfy=df.iloc[0:100,4].values
X=df.iloc[0:100,[0,2]].values
#print(X)
y=np.where(dfy=='Iris-setosa',1,-1)
ppn=Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)

