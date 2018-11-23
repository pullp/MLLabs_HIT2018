# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:56:42 2018

@author: 12181
"""
import numpy as np
import matplotlib.pyplot as plt
import logging 
import time
import struct


#x = np.arange(start, end, step)
#y = np.sin(2*np.pi*x)
#y += np.random.normal(0, noise_range, len(y))


class PCADataGenerator:
    def __init__(self, dim_count=3, data_count=100):
        x = np.arange(0, 4, 4/data_count).reshape(data_count, 1)
        y= np.sin(2*np.pi*x).reshape(data_count, 1)
        z = np.random.normal(1, 0.1, data_count).reshape(data_count, 1)
        self.data = np.concatenate((x, y, z), axis=1)
        self.dim_count = dim_count
        self.data_count = data_count
        
    '''
    return MxN array, the data count is M, the data dimension is N
    '''
    def generate(self):
        pass
    
    def rotate(self, theta):
        sin_theta  = np.sin(theta)
        cos_theta = np.cos(theta)
        rot_mat = np.array([[1, 0, 0],
                           [0, cos_theta, -sin_theta],
                           [0, sin_theta, cos_theta]])
        rot_data = []
        for ele in self.data:
            rot_data.append(rot_mat @ ele.reshape(3, 1))
        self.rot_data = np.array(rot_data).reshape(
                    self.data_count, self.dim_count
                )
        self.data = self.rot_data
        return self.rot_data
    
#    def plt_rot(self):
#        plt.plot(rot_data[:, 0], rot_data[:, 1])
        
    def plot(self):
        plt.plot(self.data[:, 0], self.data[:, 1])
        plt.show()
        
    
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
        
    def low(self, pc_count=0):
        self.principle_count = self.principle_count if pc_count==0 else pc_count
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
        return pc_eig_vecs, low_data
    
    
class picture_handle:
    def __init__(self,aim_dim):
        self.src_dim = 28
        self.aim_dim = aim_dim
        self.src_dir = "t10k-images-idx3-ubyte"
        with open(self.src_dir, 'rb') as f:
            data = f.read(16)
            des, img_nums, row, col = struct.unpack_from('>IIII', data, 0)
            train_x = np.zeros((img_nums, row * col))
            for index in range(img_nums):
                data = f.read(784)
                if len(data) == 784:
                    train_x[index, :] = np.array(
                            struct.unpack_from(
                                    '>' + 'B' * (row * col), data, 0
                                )
                            ).reshape(1, 784)
            f.close()
            self.x_matrix = train_x[0].reshape(28,28)
#            print(self.x_matrix)
            
def psnr(im1, im2):
    diff = np.abs(im1 - im2)
    rmse = np.sqrt(diff).sum()
    psnr = 20 * np.log10(255/rmse)
    return psnr

def test_pac():
    pdg = PCADataGenerator()
    pdg.generate()
    plt.title('initial data')
    pdg.plot()
#    data = pdg.get_data()
    data = pdg.rotate(np.pi * 0.25)
    plt.title('rotated data')
    pdg.plot()
    
    pca = PCA(data)
    eig_vecs, low_data = pca.low()
    plt.title('pca result')
    plt.plot(low_data[:, 0], low_data[:, 1])
    
def test_pac_picture(pc_count=5):
    
    raw_img = picture_handle(3).x_matrix
    plt.title('raw image')
    plt.imshow(raw_img)
    plt.show()
    
    for i in range(1, 28, 7) :
        plt.title('principle count:' + str(i))
        pca = PCA(raw_img, i)
        eig_vecs, low_data = pca.low()
        
        compressed_img = low_data @ eig_vecs
        plt.imshow(compressed_img)
        plt.show()
        print('psnr:' + str(psnr(raw_img, compressed_img)))
    
    
    
if __name__ == '__main__':
#    test_pac()
    test_pac_picture(20)