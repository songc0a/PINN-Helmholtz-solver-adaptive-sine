"""
@author: Chao Song
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import cmath

np.random.seed(1234)
tf.set_random_seed(1234)

fre = 6.0
PI = 3.1415926
omega = 2.0*PI*fre
niter = 100000

w0 = 3

w0_n = []
misfit = []
misfit1 = []

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, z, A, B, C, ps, m, eta, delta, layers, omega):
        
        X = np.concatenate([x, z], 1)
       
        self.iter=0
        self.start_time=0
 
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.z = X[:,1:2]

        self.ps = ps
        self.m = m
        self.eta = eta
        self.delta = delta

        self.A = A
        self.B = B
        self.C = C 

        self.layers = layers
        self.omega = omega
        
        # Initialize NN
        self.weights, self.biases, self.w0 = self.initialize_NN(layers)  

        # tf placeholders 
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        
        self.u_real_pred, self.u_imag_pred, self.u_loss, self.q_loss = self.net_NS(self.x_tf, self.z_tf)

        # loss function we define
        self.loss = tf.reduce_sum(tf.square(tf.abs(self.u_loss))) + tf.reduce_sum(tf.square(tf.abs(self.q_loss)))
        # optimizer used by default (in original paper)        
            
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        w0 = tf.Variable(tf.zeros([1,1], dtype=tf.float32)+3.0, dtype=tf.float32)
        for l in range(0,num_layers-1):

            if (l == 0):
               W = self.xavier_init_first(size=[layers[l], layers[l+1]])
            else:
               W = self.xavier_init(w0,size=[layers[l], layers[l+1]])

            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32)+0.0, dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases, w0
        
    def xavier_init(self, w0, size):
        in_dim = size[0]
        out_dim = size[1]        
        w_std = np.sqrt(1.0/in_dim)/w0

        return tf.Variable(tf.random_uniform([in_dim, out_dim], -w_std, w_std), dtype=tf.float32)

    def xavier_init_first(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        w_std = (1./np.sqrt(in_dim)) 

        return tf.Variable(tf.random_uniform([in_dim, out_dim], -w_std, w_std), dtype=tf.float32)

    
    def neural_net(self, X, weights, biases, w0):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        H = w0*H

        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, z):

        omega = self.omega
        m = self.m
        eta = self.eta
        delta = self.delta
        ps = self.ps

        A = self.A
        B = self.B
        C = self.C

        u_and_q = self.neural_net(tf.concat([x,z], 1), self.weights, self.biases, self.w0)
        u_real = u_and_q[:,0:1]
        u_imag = u_and_q[:,1:2]
        q_real = u_and_q[:,2:3]
        q_imag = u_and_q[:,3:4]

        u = tf.complex(u_real, u_imag)
        q = tf.complex(q_real, q_imag)

        dudx = fwd_gradients(u, x)
        dudz = fwd_gradients(u, z)

        dudxx = fwd_gradients((A*dudx), x)
        dudzz = fwd_gradients((B*dudz), z)  

        dqdx = fwd_gradients(q, x)
        dqdz = fwd_gradients(q, z)

        dqdxx = fwd_gradients((A*dqdx), x)
        dqdzz = fwd_gradients((B*dqdz), z)


        u_loss =  C*omega*omega*m*u + dudxx + dqdxx + dudzz/(1+2*delta) - ps #  L u - f
        q_loss =  C*omega*omega*m*q + 2*eta*(dudxx + dqdxx) 
    
        return u_real, u_imag, u_loss, q_loss        

    def callback(self, loss):
        #print('Loss: %.3e' % (loss))
        misfit1.append(loss)
        self.iter=self.iter+1
        if self.iter % 10 == 0:
                elapsed = time.time() - self.start_time
                print('It: %d, LBFGS Loss: %.3e,Time: %.2f' %
                      (self.iter, loss, elapsed))
                self.start_time = time.time()
    
      
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.z_tf: self.z}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            misfit.append(loss_value)         
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                [loss_value, w0_value] = self.sess.run([self.loss, self.w0], tf_dict)
                w0_n.append(w0_value)
                print('It: %d, Loss: %.3e,Time: %.2f,w0:%.2f' %
                      (it, loss_value, elapsed,w0_value))

                start_time = time.time()

            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

            
    
    def predict(self, x_star, z_star):
        
        tf_dict = {self.x_tf: x_star, self.z_tf: z_star}
        
        u_real_star = self.sess.run(self.u_real_pred, tf_dict)
        u_imag_star = self.sess.run(self.u_imag_pred, tf_dict)

        return u_real_star, u_imag_star
        
        
if __name__ == "__main__": 
       
    layers = [2, 40, 40, 40, 40, 40, 40,40, 40, 4]
    
    # Load Data
    data = scipy.io.loadmat('vti_scatter_6Hz_sine.mat')
           
    u_real = data['U_real'] 
    u_imag = data['U_imag'] 

    ps = data['Ps'] 

    x = data['x'] 
    z = data['z'] 
    m = data['m'] 
    eta = data['eta']
    delta = data['delta']


    A = data['A'] 
    B = data['B'] 
    C = data['C'] 
    
    N = x.shape[0]
    N_train = N
    # Training Data    
    idx = np.random.choice(N, N_train, replace=False)

    x_train = x[idx,:]
    z_train = z[idx,:]

    ps_train = ps[idx,:]
    A_train = A[idx,:]
    B_train = B[idx,:]
    C_train = C[idx,:]

    m_train = m[idx,:]
    eta_train = eta[idx,:]
    delta_train = delta[idx,:]
    # Training
    model = PhysicsInformedNN(x_train, z_train, A_train, B_train, C_train,  ps_train, m_train, eta_train, delta_train, layers, omega)
    model.train(niter)

    scipy.io.savemat('w0_scatter_vti.mat',{'w0':w0_n})

    scipy.io.savemat('loss_adam_sine.mat',{'misfit':misfit})
    scipy.io.savemat('loss_lbfgs_sine.mat',{'misfit1':misfit1})
    
    
    # Test Data

    x_star = x
    z_star = z

    u_real_star = u_real
    u_imag_star = u_imag


    # Prediction
    u_real_pred, u_imag_pred = model.predict(x_star, z_star)
    
    # Error
   
    scipy.io.savemat('u_real_pred_sine_vti.mat',{'u_real_pred':u_real_pred})
    scipy.io.savemat('u_imag_pred_sine_vti.mat',{'u_imag_pred':u_imag_pred})

    #scipy.io.savemat('u_real_star_10hz.mat',{'u_real_star':u_real_star})
    #scipy.io.savemat('u_imag_star_10hz.mat',{'u_imag_star':u_imag_star})

