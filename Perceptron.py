#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
class Perceptron(object):
    def __init__(self,n_eta,n_iter):
        self.n_eta=n_eta
        self.n_iter=n_iter
    def net_input_output(self,X):
        return np.dot(X,w[1:])+w[0]
    def predict(self,X):
        return np.where(self.net_input_ouput(X)>=0,1,-1)
    def w_update(self,X,y):
        self.w=np.zeros(1+X.shape[1])
        self.error=[]
        for i in range(self.n_iter):
            for xi,yi in zip(X,y):
                wupdate=self.n_eta*(yi-self.predict(xi))
                self.w[1:]+=wupdate*xi
                self.w[0]+=wupdate
            return self
        
            
            
        
        
        

