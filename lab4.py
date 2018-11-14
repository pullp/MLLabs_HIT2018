# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:56:42 2018

@author: 12181
"""
import numpy as np
import matplotlib.pyplot as plt
import logging 
import time
import heapq


#x = np.arange(start, end, step)
#y = np.sin(2*np.pi*x)
#y += np.random.normal(0, noise_range, len(y))


class PCADataGenerator:
    def __init__(self, dim_count=3, data_count=100):
        x = np.arange(1, 10, 9/data_count).reshape(data_count, 1)
        y= np.sin(2*np.pi*x).reshape(data_count, 1)
        z = np.ones(data_count).reshape(data_count, 1)
        self.data = np.concatenate((x, y, z), axis=1)
        
    '''
    return MxN array, the data count is M, the data dimension is N
    '''
    def generate(self):
        pass
    
    def get_data(self):
        return self.data
    
class PCA:
    '''
    data is MxN array, the data count is M, the data dimension is N
    '''
    def __init__(self, data, principle_count=2):
        self.data = data
        self.pc = principle_count
        self.data_count, self.dim_count = data.shape
        self.principle_count = principle_count
        
    def low(self):
        mean_vals = self.data.mean(axis=0)
        mean_removed = np.array([row - mean_vals for row in self.data])
        cov = mean_removed.T @ mean_removed / (self.dim_count - 1)
        eig_vals, eig_vecs = np.linalg.eig(cov)
#        nlargets_idx = heapq.nlargest(self.principle_count, eig_vals, eig_vals.take)
        sorted_idx = np.argsort(eig_vals)
        nlargets_idx = sorted_idx[:-(self.principle_count+1):-1]
#        pc_eig_vecs = eig_vecs[:, nlargets_idx]
        pc_eig_vecs = []
        for idx in nlargets_idx:
            pc_eig_vecs.append(eig_vecs[idx])
        pc_eig_vecs = np.array(pc_eig_vecs)
        
        low_data = self.data @ pc_eig_vecs.T
        
        return low_data
    
    
        
        
def test_pac():
    pdg = PCADataGenerator()
    pdg.generate()
    data = pdg.get_data()
    
    pca = PCA(data)
    
    low_data = pca.low()
    plt.plot(low_data[:, 0], low_data[:, 1])
    
    
if __name__ == '__main__':
    test_pac()