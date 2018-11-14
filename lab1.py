yiimport numpy as np
from numpy import matmul, array
import matplotlib.pyplot as plt
import time

start, end, step = -1, 1, 0.01
freq = 2*np.pi
noise_range = 0.4


# generate dots
x = np.arange(start, end, step)
y = np.sin(2*np.pi*x)
y += np.random.normal(0, noise_range, len(y))
plt.plot(x, y, "r+")

target = np.mat(array(y).reshape(len(y), 1))


class GradDec:
    #data format [[[x1, x2, x3], y1], [[x21, x22, x23] y2]]
    data
    def __init__(self, data, dim=1, max_iter = 60000, min_loss_gap =1e-4, reg=0):
        self.data = data
        self.reg = reg
        self.max_iter = max_iter
        self.dim = 1
        
    def getw(self):
        current_iter = 0
        w = np.random.normal(size=self.dim).reshape(m, 1)
        grad = getgrad(w)
        loss = getloss(w)
        loss_gap = 100000
        gamma = 1e-3
        while(loss_gap > self.min_loss_gap and current_iter < self.max_iter):
            current_iter += 1
            grad = getgrad(w)
            if(loss_gap < 1e-3):
                gamma = recalc_gamma()
                gamma = 1e-3
            w -= gamma * grad
            new_loss = self.getloss(w)
            loss_gap = new_loss - loss
            while(loss_gap < 0):
                gamma = 0.9 *gamma
                new_loss = self.getloss(w)
                loss_gap = new_loss - loss
            loss = new_loss
        print(w)
            
        

    def getgrad(self, currentW):
        
    def getloss(self, currentW,reg=0):
        
        

def loss_reg(matrix, w, t, l):
    temp = matmul(matrix, w) -t
    return 0.5*(matmul(temp.T, temp) + l*matmul(w.T, w)).getA()[0][0]


def loss(matrix, w, t):
    return loss_reg(matrix, w, t, l=0)


    

def matmutimul(*args):
    ret = np.eye(args[0].shape[0])
    for ele in args:
        ret = matmul(ret, ele)
    return ret


        
def calc_reg(m=10, l=1):
    print("解析解法: m=%d, 惩罚系数=%f"%(m, l))
    matrix = np.mat(array([[i**j for j in range(0, m)] for i in x]))
    w = matmul(matmul(target.T, matrix), (matmul(matrix.T, matrix) + np.eye(m)*l).I).T
    out = matmul(matrix, w)
#    print(w)
    plt.plot(x, out)
    loss = loss_reg(matrix, w, target, l)
    return loss
    
    
def calc(m=10):
    return calc_reg(m, l=0)


def recalc_gama(matrix, w, grad):
    ret = 0
    min_loss = 1e100
    for i in range(1, 10):
        gama_cur = 10 ** (-i)
        w_new = w - gama_cur*grad
        lo = loss(matrix, w_new, target)
        if(lo < min_loss):
            min_loss = lo
            ret = gama_cur
    return ret*10
    
def grad_dec_reg(m=5, gama = 1e-3, l = 1e-5):
    print("梯度下降法: m=%d, 学习率=%f, 惩罚系数=%f"%(m, gama, l))
    matrix = array([[i**j for j in range(0, m)] for i in x])
    w = np.random.normal(size=m).reshape(m, 1)
#    A = matmul(matrix.T, matrix) + l * np.eye(m)
    A = matrix
#    print(A.shape)
#    b = matmul(matrix.T, target)
    b = target
#    print(w.T)
    lo = loss(matrix, w, target)
    lo_prev = lo + 10
    loop = 0
    grad = matmul(A.T,matmul(A, w)-b)
    while(abs(lo - lo_prev) > 1e-4):
        grad = (A.T@(A@w -b) + l*w)/len(y)
#        print(grad.shape)
        if(abs(lo - lo_prev) < 1e-4):
            gama = recalc_gama(matrix, w, grad)
#            gama = 1e-3
#            print("larger gama:" + str(gama))
        sub = gama * grad
#        print(sub.shape)
        w -= sub
#        print(grad)
        lo_prev = lo
        lo = loss(matrix, w, target)
        while(lo >= lo_prev):
            w += gama*grad
            gama = 0.99*gama
            w -= gama*grad
            lo = loss(matrix, w, target)
        print("\rloss = " + str(lo) + " loop:" +str(loop), end='')
        loop += 1
#        print()
    out = matmul(matrix, w)
#    print(w)
    plt.plot(x, out)
    return lo
    
def grad_dec(m=4, gama = 1e-3):
    return grad_dec_reg(m, gama, l=0)


    
def con_grad(m=10):
    return con_grad_reg(m, l=0)
    
def con_grad_reg(m=10, l=1e-1):
    print("共轭梯度法: m=%d, 惩罚系数=%f"%(m, l))
    matrix = np.mat(array([[i**j for j in range(0, m)] for i in x]))
    w = (np.random.normal(size=m)).reshape(m, 1)
    A = (matrix.T @ matrix) + l * np.eye(m)
    b = (matrix.T @ target)
    w_array = [0 for i in range(0, 2*m)]
    r_array = [0 for i in range(0, 2*m)]
    p_array = [0 for i in range(0, 2*m)]
    w_array[0] = w
    r_array[0] = b - A@w
    p_array[0] = r_array[0]
    for k in range(0, m):
        a_k = (r_array[k].T @ r_array[k]).getA()[0][0] / (p_array[k].T @ A @ p_array[k]).getA()[0][0]
#        print(a_k)
#        print(w_array[k].shape, (a_k * p_array[k]).shape)
        w_array[k+1] = w_array[k] + a_k*p_array[k]
        r_array[k+1] = r_array[k] - a_k * (A @ p_array[k])
        b_k = (r_array[k+1].T @ r_array[k+1]).getA()[0][0] / (r_array[k].T @ r_array[k]).getA()[0][0]
        p_array[k+1] = r_array[k+1] + b_k * p_array[k]
        w = w_array[k+1]
#        print(w_array[k+1])
#        print(loss(matrix, w, target))
    plt.plot(x, (matrix @ w))
    loss = loss_reg(matrix, w, target, l)
    return loss
    
        
        
        
   
if __name__ == "__main__":
    print("y = sin(%dx), x's range = [%d, %d], points = %d" %(freq, start, end, (end-start)/step))
    print("噪声方差: %f"%(noise_range))
    time_start = time.time()
    m = 30
    l= 1e-3
#    lo = calc_reg(10, 1e-3)
    lo = calc(40)
#    lo = grad_dec(m=20, gama=1)
#    lo = grad_dec_reg(m=20,gama = 1, l=1e-3)
#    lo = con_grad(m=20)
#    lo = con_grad_reg(m=20, l=1e-3)
    time_end = time.time()
    print("use time:", time_end - time_start, "s")
    print("loss = ", lo)
    


#def error(w)
    