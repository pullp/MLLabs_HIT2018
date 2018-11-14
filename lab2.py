# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:11:37 2018

@author: 12181
"""

# 梯度下降法还差正则项
# 生成数据
# 牛顿法
import numpy as np
from numpy import matmul, array
import matplotlib.pyplot as plt
import time


class DataGenerator:
    #default dimension is 2, independent
    def __init__(self, size=50, indp = True, same_sigma=True, dim=2, feature = 2):
        self.dim = dim
        self.feature = feature
        self.indp = indp
        self.size = size
        self.pltdata = []
        self.same_sigma = same_sigma
        self.left = 1000000
        self.right = -1000000 #boundry
        self.isfile = False
    #generator and store data, then return data
    #data format [[[1, x11, x12, x13], y1], [[1, x21, x22, x23] y2]]
    def generate(self):
        data = []
        o = np.ones((self.size, 1))
        for i in range(0, self.feature):
            group = []
            if(self.indp):
                cov = np.eye(self.dim) * 1
            else:
                cov = np.array([[10, -5], [-5, 10]])#todo find a positive definite matrix
            sigma = cov if self.same_sigma else (cov * (i+1)*5)
            group = np.random.multivariate_normal(i * np.ones(self.dim) * 6, sigma , self.size)
            self.left = self.left if self.left <= group.min() else group.min()
            self.right = self.right if self.right >= group.max() else group.max()
            self.pltdata.append(group)
            group = np.concatenate((o, group), axis = 1)
            for ele in group:
                data.append([ele.tolist(), i])
        self.data = data
        return data
    
    def fromfile(self, filename='./iris.data.txt'):
        content = open(filename, 'r').read()
        datas = content.split('\n')
#        self.pltdata = [[], []]
        plt_groups = [[], []]
        data = []
        for ele in datas:
            bricks = ele.split(',')
            group_type = int(bricks[2])
            x = float(bricks[0])
            self.left = self.left if self.left <= x else x
            self.right = self.right if self.right >= x else x
            y = float(bricks[1])
            plt_groups[group_type].append([x, y])
            data.append([[1, x, y], group_type])
        self.data = data
        for group in plt_groups:
            self.pltdata.append(np.array(group))
        self.isfile = True
        return data
        
        
    def idx2color(self, idx):
        if idx == 0:
            return 'r+'
        elif idx == 1:
            return 'g+'
        else:
            return 'b+'
        
    def printinfo(self):
        if self.isfile:
            print('来自文件')
            return
        info = '上图: '
        if self.indp:
            info += '独立, '
        else:
            info += '不独立, '
        if self.same_sigma:
            info += '相同的sigma'
        else:
            info += '不同的sigma'
        print(info)
    def plot2(self):
        
        if not self.dim  == 2:
            print("wrong dimension")
            return
        idx = 0
        for group in self.pltdata:
            plt.plot(group[:,0], group[:,1], self.idx2color(idx))
            idx += 1
        plt.show()
        self.printinfo()
#        plt.plot(data[:][0][1], data[:][0][2], self.idx2color(data[:][1]))
        
        #line = [w0, w1, w2] => w0 + w1 x + w2 y = 0
    def plot2line(self, line):
        
        x1 = self.left - 1
        x2 = self.right + 1
        y1 = -(line[0] + line[1]*x1) / line[2]
        y2 = -(line[0] + line[1]*x2) / line[2]
#        print('x1, y1', x1, y1)
#        print('x2, y2', x2, y2)
        plt.plot([x1, x2], [y1, y2])
        idx = 0
        for group in self.pltdata:
            plt.plot(group[:,0], group[:,1], self.idx2color(idx))
            idx += 1
        plt.plot(3, 3, 'b+')
        plt.show()
        self.printinfo()
            
    #return data generated last time
    def samedata(self):
        return self.data
        
    def test(self, size = 10):
        ret = []
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(10, 1, 100)
        for ele in data1:
            ret.append([[1, ele], 0])
        for ele in data2:
            ret.append([[1, ele], 1])
        return ret
        
        
        
class GradDec:
    #data format [[[1, x11, x12, x13], y1], [[1, x21, x22, x23] y2]]
    def __init__(self, data, dim=2, max_iter = 60000, min_loss_gap =1e-4, reg=0):
        self.data = data
        self.reg = reg
        self.max_iter = max_iter
        self.min_loss_gap = min_loss_gap
        self.dim = dim+1
        
    def getw(self):
        current_iter = 0
        target = np.random.normal(size=self.dim).reshape(self.dim, 1)
        grad = self.getgrad(target)
        loss = self.getloss(target)
        loss_gap = -100000
        gamma = 1e-3
        while(-loss_gap > self.min_loss_gap and current_iter < self.max_iter):
            current_iter += 1
            grad = self.getgrad(target)
            if(loss_gap < 1e-3):
#                gamma = recalc_gamma()
                gamma = 1e-3
#            print(target.shape)
            temp = gamma * grad
            target = target - temp
            new_loss = self.getloss(target)
            loss_gap = new_loss - loss
            while(loss_gap > 0 and current_iter < self.max_iter):
                current_iter += 1
                gamma = 0.6 *gamma
                target = target - gamma*grad
                new_loss = self.getloss(target)
                loss_gap = new_loss - loss
            loss = new_loss
            print("\rloss = " + str(loss) + " loop:" +str(current_iter) + ';', end='')
#            print("loss = " + str(loss) + " loop:" +str(current_iter))
        return target

    def getgrad(self, target):
        grad = 0
        for ele in self.data:
            y = ele[1]
            x = ele[0]
            temp = np.exp(target.T @ x)
            grad += x *(y - temp/(1+temp))
        return (grad.reshape(self.dim, 1)  + self.reg * target) * (-1)
        
    def getloss(self, target):
        loss = 0
        for ele in self.data:
            y = ele[1]
            x = ele[0]
            temp = (target.T @ x)
            loss += y *temp - np.log(1 + np.exp(temp))
        reg = self.reg * (target.T @ target)[0][0]
        return loss[0] * (-1) + reg
    
        
class Newton:
    def __init__(self, data, dim=2, max_iter = 600, min_loss_gap =1e-5, reg=0):
        self.data = data
        self.reg = reg
        self.max_iter = max_iter
        self.min_loss_gap = min_loss_gap
        self.dim = dim+1
    
    def getw(self):
        current_iter = 0
        target = np.random.normal(size=self.dim).reshape(self.dim, 1)
        gamma = 0.01
        while(current_iter < self.max_iter):
            current_iter += 1
            target -= gamma * self.getdec(target)
            loss = self.getloss(target)
            print("\rloss = " + str(loss) + " loop:" +str(current_iter), end='')
        return target
        
    def getgrad(self, target):
        grad = np.zeros(self.dim)
        for ele in self.data:
            y = ele[1]
            x = ele[0]
            temp = np.exp(target.T @ x)
            grad += x *(y - temp/(1+temp))
        return (grad.reshape(self.dim, 1)  + self.reg * target) * (-1)

    def getdec(self, target):
        grad = self.getgrad(target)
        hession = self.gethession(target)
        ret = np.linalg.inv(hession) @ grad
        return ret
        
    def gethession(self, target):
        hession = np.zeros((self.dim, self.dim))
        for ele in self.data:
            x = ele[0]
            Al = np.exp(target.T @ x)[0]
            temp = Al / ((1 + Al)**2) 
            for i in range(0, self.dim):
                for j in range(0, self.dim):
                    hession[i][j] -= x[i] * x[j] * (temp + (self.reg if i==j else 0))
        return (hession)* (-1)
        
    def getloss(self, target):
        loss = 0
        for ele in self.data:
            y = ele[1]
            x = ele[0]
            temp = (target.T @ x)
            loss += y *temp - np.log(1 + np.exp(temp))
        reg = self.reg * (target.T @ target)[0][0]
        return loss[0] * (-1) + reg
    
def test_dec():
    print('test grad dec')
#    DG = DataGenerator()
##    data = DG.test()
#    data = DG.generate()
#    DG.plot2()
##    print(data)
#    GD = GradDec(data)
#    w = GD.getw()
#    print(w, w[0]/w[1])
    dg1, dg2, dg3, dg4 = test_dg()
    gd1 = GradDec(dg1.data, reg = 0.1)
    gd2 = GradDec(dg1.data, reg = 0.01)
    gd3 = GradDec(dg1.data, reg = 0.001)
    gd4 = GradDec(dg1.data, reg = 0.0001)
#    
    w1 = gd1.getw()
    dg1.plot2line(w1)
    
    w2 = gd2.getw()
    dg1.plot2line(w2)
    w3 = gd3.getw()
    dg1.plot2line(w3)
    w4 = gd4.getw()
    dg1.plot2line(w4)
    
    
def test_newton():
    print('test Newton')
    DG = DataGenerator()
#    data = DG.test()
    data = DG.generate()
    DG.plot2()
    
    dg1, dg2, dg3, dg4 = test_dg()
    nt1 = Newton(dg1.data, reg=1e-1)
#    nt2 = Newton(dg1.data, reg=)
    nt2 = Newton(dg1.data, reg=1e-2)
    nt3 = Newton(dg1.data, reg=1e-3)
    nt4 = Newton(dg1.data, reg=1e-4)
    
    w1 = nt1.getw()
    dg1.plot2line(w1)
    print('reg = ' + str(nt1.reg))
    
    w2 = nt2.getw()
    dg1.plot2line(w2)
    print('reg = ' + str(nt2.reg))
    
    w3 = nt3.getw()
    dg1.plot2line(w3)
    print('reg = ' + str(nt3.reg))
    
    w4 = nt4.getw()
    dg1.plot2line(w4)
    print('reg = ' + str(nt4.reg))
    
def test_dg():
    dg1 = DataGenerator(indp=True, same_sigma=True)
    dg1.generate()
    dg1.plot2()
    
    dg2 = DataGenerator(indp=True, same_sigma=False)
    dg2.generate()
    dg2.plot2()
    
    dg3 = DataGenerator(indp=False, same_sigma=True)
    dg3.generate()
    dg3.plot2()
    
    dg4 = DataGenerator(indp=False, same_sigma=False)
    dg4.generate()
    dg4.plot2()
    
    return dg1, dg2, dg3, dg4
    
def test_uci():
    dg = DataGenerator()
    data = dg.fromfile()
    dg.plot2()
    nt = Newton(data)
    wn = nt.getw()
    dg.plot2line(wn)
    print('Newton')
    
    gd = GradDec(data)
    wg = gd.getw()
    dg.plot2line(wg)
    print('grad dec')
    

def plt_test():
    pass
    
if __name__ == '__main__':
#    test_dg()
    time_start = time.time()
#    test_dg()
    test_uci()
#    test_newton()
#    test_dec()
    time_end = time.time()
    print("\nuse time:", time_end - time_start, "s")
#    test_dec()
#    test_dg()
    
    