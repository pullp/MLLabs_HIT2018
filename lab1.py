import numpy as np
from numpy import matmul, array
import matplotlib.pyplot as plt
import time

start, end, step = -1, 1, 0.05
noise_range = 0.1


# generate dots
x = np.arange(start, end, step)
y = np.sin(3*np.pi*x)
y += np.random.normal(0, noise_range, len(y))
plt.plot(x, y, "r+")

target = np.mat(array(y).reshape(len(y), 1))


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
    matrix = np.mat(array([[i**j for j in range(0, m)] for i in x]))
    w = matmul(matmul(target.T, matrix), (matmul(matrix.T, matrix) + np.eye(m)*l).I).T
    out = matmul(matrix, w)
    print(w)
    plt.plot(x, out)
    
def calc(m=10):
    calc_reg(m, l=0)


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
    return ret
    
def grad_dec_reg(m=5, gama = 1e-3, l = 1e-5):
    matrix = np.mat(array([[i**j for j in range(0, m)] for i in x]))
    w = np.mat(np.random.normal(size=m)).T
    A = matmul(matrix.T, matrix) + l * np.eye(m)
    b = matmul(matrix.T, target)
#    print(w.T)
    lo = loss(matrix, w, target)
    lo_prev = lo + 10
    loop = 0
    grad = 2*matmul(A.T,matmul(A, w)-b)
    while(abs(lo - lo_prev) > 1e-7):
        if(abs(lo - lo_prev) < 1e-4):
            gama = recalc_gama(matrix, w, grad)
            print("larger gama:" + str(gama))
        grad = 2*matmul(A.T,matmul(A, w)-b)
        w -= gama*grad
        lo_prev = lo
        lo = loss(matrix, w, target)
        while(lo >= lo_prev):
            w += gama*grad
            gama = 0.99*gama
#            print("new gama:"+str(gama))
            w -= gama*grad
            lo = loss(matrix, w, target)
#        gama = 1e-3
        print("\rloss = " + str(lo) + " loop:" +str(loop), end='')
        loop += 1
#        print()
    out = matmul(matrix, w)
#    print(w)
    plt.plot(x, out)
    
def grad_dec(m=4, gama = 1e-3):
    grad_dec_reg(m, gama, l=0)


    
def con_grad(m=10):
    con_grad_reg(m, l=0)
    
def con_grad_reg(m=10, l=1e-1):
    matrix = np.mat(array([[i**j for j in range(0, m)] for i in x]))
    w = np.mat(np.random.normal(size=m)).T
    A = matmul(matrix.T, matrix) + l * np.eye(m)
    b = matmul(matrix.T, target)
    w_array = [0 for i in range(0, 2*m)]
    r_array = [0 for i in range(0, 2*m)]
    p_array = [0 for i in range(0, 2*m)]
    w_array[0] = w
    r_array[0] = b-matmul(A, w)
    p_array[0] = r_array[0]
    for k in range(0, m):
        a_k = matmul(r_array[k].T, r_array[k]).getA()[0][0] / matmutimul(p_array[k].T, A, p_array[k]).getA()[0][0]
#        print(a_k)
#        print(w_array[k].shape, (a_k * p_array[k]).shape)
        w_array[k+1] = w_array[k] + a_k*p_array[k]
        r_array[k+1] = r_array[k] - a_k*matmul(A, p_array[k])
        b_k = matmul(r_array[k+1].T, r_array[k+1]).getA()[0][0] / matmul(r_array[k].T, r_array[k]).getA()[0][0]
        p_array[k+1] = r_array[k+1] + b_k * p_array[k]
        w = w_array[k+1]
#        print(w_array[k+1])
#        print(loss(matrix, w, target))
    plt.plot(x, matmul(matrix, w))
    
        
        
        
   
if __name__ == "__main__":
    time_start = time.time()
    m = 30
    l= 1e-3
#    calc_reg(20, 1e-5)
#    calc(20)
#    grad_dec(m=20, gama=1e-1)
#    grad_dec_reg(gama = 1e-3)
#    con_grad(m=20)
    con_grad_reg(m=20, l=1e-5)
    time_end = time.time()
    print("use time:", time_end - time_start)
    


#def error(w)
    