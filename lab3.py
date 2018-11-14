# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:57:53 2018

@author: 12181
"""
import numpy as np
import matplotlib.pyplot as plt
import logging 
import random
import time
from scipy.stats import multivariate_normal


'''
problems:
    1. 不同类型的点的数据量如果差的太多会导致分不出来, 因为差的太多会导致重新计算
        中心点的时候出现偏差
    2. 如果两类数据靠的太近也可能分不出来
    
    
'''
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

log = logging.getLogger('lab3')
log.setLevel(logging.INFO)


class  DataGenerator:
    def __init__(self, type_count=2, size=10):
        mu_differ = 3
#        cov_differ = 0
        mu = np.array([1, 1])
        cov = np.array([[1, 0], [0, 1]]) * 0.1
        self.mus = []
        for i in range(type_count):
            self.mus.append((mu*(i+1)*mu_differ).tolist())
            
        self.data = np.random.multivariate_normal(self.mus[0], cov, size)
        for i in range(1, type_count):
            group = np.random.multivariate_normal(self.mus[i], cov*(i+1), size)
            self.data = np.concatenate((self.data, group))
        
    def plot(self):
        plt.plot(self.data[:, 0], self.data[:, 1], 'r+')
        
    def get_data(self):
        return self.data
    
    def get_centers(self):
        return self.mus


class ClusterBasic:
    def __init__(self, data, type_count=2):
        self.type_count = type_count
        self.data = data.tolist()
        self.data_count = len(self.data)
        self.clus = []
        for i in range(type_count):
            self.clus.append([])
        log.info('initializing....')
        log.debug('data is:\n'+str(data))
        self._init_theta()
        self._init_clus()
        
        
    def _init_theta(self):
        pass
    
    def _init_clus(self):
        pass
    
    def cluster(self):
        pass
    
    def plot(self, title='default'):
        plt.title(title)
        for i in range(self.type_count):
            group = self.clus[i]
            if(len(group) == 0):
                continue
            plt_array = np.array(group)
            plt.plot(plt_array[:, 0], plt_array[:, 1], '+')
        for i in range(self.type_count):
            center = self.mus[i]
            plt.plot(center[0], center[1], 'k*')
        plt.show()
    

    
        
class Kmeans(ClusterBasic):
    def __init__(self, data, type_count=2):
        ClusterBasic.__init__(self, data, type_count)
        self.has_dynamic_point = True
        
    def _init_theta(self):
        self.mus = random.sample(self.data, self.type_count)
        log.info('initialize centers: ' + str(self.mus))
    
    def _get_distance(self, p1, p2):
        return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))
        
    def _closest_center_idx(self, point):
        log.debug('point: '+str(point))
        idx = 0
        cur_center = self.mus[0]
        min_dis = self._get_distance(point, cur_center)
        log.debug('dis 1: ' + str(min_dis)+ ' center: '+str(cur_center))
        for i in range(1, self.type_count):
            cur_center = self.mus[i]
            cur_dis = self._get_distance(point, cur_center)
            log.debug('dis 2: ' + str(cur_dis)+ ' center: '+str(cur_center))
            if(cur_dis < min_dis):
                min_dis = cur_dis
                idx = i
        log.debug(' index is ' + str(idx))
        return idx

    def _init_clus(self):
        for group in self.clus:
            group.clear()
        log.info('initialize clus')
        for ele in self.data:
            closest_idx = self._closest_center_idx(ele)
            self.clus[closest_idx].append(ele)
        for i in range(self.type_count):
            log.debug('type %d, has %d points'%(i, len(self.clus[i])))
    
    def _recalc_centers(self):
        for i in range(self.type_count):
            if(len(self.clus[i]) == 0):
                continue
            self.mus[i] = np.array(self.clus[i]).mean(0).tolist()
        log.debug('recalc new centers:' + str(self.mus))
#        log.debug(self.mus)
            
    def set_centers(self, centers):
        self.mus = centers
        log.info('set centers to: ' + str(self.mus))
        self._init_clus() # different center contribute to different clusster
        self.plot()
        
    def cluseter(self):
        current_iter = 0
        dynamic_count = 0
        while(self.has_dynamic_point and current_iter < self.max_iter):
            log.debug('loop:' + str(current_iter))
            self.plot('loop:' + str(current_iter))
            current_iter += 1
            self.has_dynamic_point = False
            self._recalc_centers()
            for i in range(self.type_count):
                group = self.clus[i]
                for point in group:
                    new_idx = self._closest_center_idx(point)
                    if not new_idx == i:
                        dynamic_count += 1
                        log.debug('new dynamic point(total' + str(dynamic_count) + ':' + str(point))
                        self.has_dynamic_point = True
                        self.clus[i].remove(point)
                        self.clus[new_idx].append(point)
            
            
class GmmDataGenerator(DataGenerator):
    def __init__(self, 
                 mus=[[1,1], [1,9], [9,9]], 
                 covs=[
                        [[1,0],[0,1]], 
                        [[2,-1],[-1,2]],
                        [[1,0],[0,1]]
                 ],
                 probs=[0.4, 0.2, 0.4] ,
                 size=1000):
        type_count = len(mus)
        if(len(covs) != type_count or len(probs) != type_count):
            log.error('size should be equal')
        self.probs = []
        total_prob = 0
        for p in probs:
            total_prob += p
        for p in probs:
            self.probs.append(p/total_prob)
        self.mus = mus
        self.covs = covs
        
        self.data = np.random.multivariate_normal(mus[0], covs[0], int(size*self.probs[0]))
        for i in range(1, type_count):
            group = np.random.multivariate_normal(mus[i], covs[i], int(size*self.probs[i]))
            self.data = np.concatenate((self.data, group))
            
            
class EM(ClusterBasic):
    def __init__(self, data, type_count=3, max_iter=20):
        ClusterBasic.__init__(self, data, type_count)
        self.max_iter = max_iter
        self.data = np.array(self.data)
        
    def _init_theta(self):
        self.mus = random.sample(self.data, self.type_count)
        self.covs = []
        self.prevs = []
        for i in range(self.type_count):
            self.covs.append(np.eye(2))
            self.prevs.append(1.0/self.type_count)
        self.mus = np.array(self.mus)
        self.covs = np.array(self.covs)
        self.prevs = np.array(self.prevs)
        
    def _init_clus(self):
        self.clus = [[] for i in range(self.type_count)]
#        data_count = len(self.data)
        probs = np.array([ [0.0 for j in range(0, self.type_count)] 
                    for i in range(len(self.data))] )
                
        for i in range(self.data_count):
            sum_temp = 0
            for j in range(0, self.type_count):
                pdf_j = multivariate_normal.pdf(
                        self.data[i], mean=self.mus[j], cov=self.covs[j]
                    )
                temp= self.prevs[j] * pdf_j
                probs[i][j] = temp
                sum_temp += temp
            probs[i] /= sum_temp
                    
        for i in range(self.data_count):
            idx = probs[i].argmax()
            self.clus[idx].append(self.data[i])
        
    
    
    def _get_prob(self, point, mu, cov):
        pass        
        
    def cluster(self):
        self.probs = np.array([ [0.0 for j in range(0, self.type_count)] 
                    for i in range(self.data_count)] )
        # todo move to init_clus
        data_count = self.data_count
        current_iter= 0
        while current_iter < self.max_iter:
            self.plot('loop:' + str(current_iter))
            self.clus = [[] for i in range(self.type_count)]
            current_iter += 1
            #E step
            for i in range(data_count):
                sum_temp = 0
                for j in range(0, self.type_count):
                    pdf_j = multivariate_normal.pdf(
                                self.data[i], mean=self.mus[j], cov=self.covs[j]
                        )
                    temp = self.prevs[j] * pdf_j
                    self.probs[i][j] = temp
                    sum_temp += temp
                self.probs[i] /= sum_temp
            # M step
            # get new mu
            new_mus_up = [np.zeros(2) for i in range(self.type_count)]
            N_k = [0.0 for i in range(self.type_count)]
            for i in range(data_count):
                data = self.data[i]
                # generate clus with respect to probs
                idx = self.probs[i].argmax()
                self.clus[idx].append(data)
                for j in range(0, self.type_count):
                    N_k[j] += self.probs[i][j]
                    new_mus_up[j] += self.probs[i][j] * data
                    
            total_prevs = sum(N_k)
            for j in range(0, self.type_count):
                self.mus[j] = new_mus_up[j] / N_k[j]
                 # get new prev
                self.prevs[j] = N_k[j] / total_prevs 
            
            # get new cov
            new_covs_up = [
                    np.zeros(4).reshape(2, 2) for i in range(self.type_count)
                ]
            for i in range(data_count):
                data = self.data[i]
                for j in range(0, self.type_count):
                    tmp = (data - self.mus[j]).reshape(2, 1)
                    new_covs_up[j] += self.probs[i][j] * (tmp @ tmp.T)
                    
            for j in range(0,self.type_count):
                self.covs[j] = new_covs_up[j]  / N_k[j]
                
            


def test_em():
    gdg = GmmDataGenerator(size = 300)
    gdg.plot()
    
    em = EM(data = gdg.data, type_count=3)
    em.cluster()

    
def test_gmmdg():
    gdg = GmmDataGenerator()
    gdg.plot()
    

def test_dg():
    dg = DataGenerator(size=1000)
    dg.plot()
    


def test_kmeans():
    type_count = 2
    dg = DataGenerator(type_count = type_count, size = 1000)
    data = dg.get_data()

    log.info('============k1==========')
    k1 = Kmeans(data, type_count = type_count)
    k1.cluseter()
    k1.plot()
    
    
if __name__ == '__main__':
    start = time.time()
#    test_gmmdg()
#    test_kmeans()
    test_em()
    
#    test_sqrt()
    end = time.time()
    log.info('time usage: ' + str(end - start))
    