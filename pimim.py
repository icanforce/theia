import numpy as np
import tensorflow as tf
import os
import pickle
#-----------------------------------------------------------------------------------------------------
lambda1 = 1#50
alpha = 1
beta = 0.5
C1 = 3
C2 = 3


class Trainer:
    def __init__(self, params):
        self.params = params    

#-----------------------------------------------------------------------------------------------------    
    def prob_I_Phi(self, UV_T, I_Phi, alpha):
        # p = tf.sigmoid( tf.multiply(UV_T, I_Phi) - alpha )
        # p = tf.sigmoid( UV_T - alpha )
        
        cond = tf.multiply(UV_T, I_Phi) > alpha
        a = tf.sigmoid( UV_T )
        b = 1 - a
        p = tf.where(cond,a,b)        
        return p
#------------------
    def prob_I_Omega(self, VV_T, I_Omega, beta, N):
        # p = tf.sigmoid( VV_T - beta )
        
        cond = tf.multiply(VV_T, I_Omega) > beta
        # print cond.get_shape()
        # a = tf.nn.relu( tf.multiply(VV_T, I_Omega) - beta )
        a = tf.sigmoid( VV_T )
        b = tf.ones([N, N], tf.float32)
        p = tf.where(cond,a,b)
        return p
#-----------------------------------------------------------------------------------------------------        
    def train(self):
        M = self.params['M']
        N = self.params['N']            
        K = self.params['K']
        mRNA_var = self.params['y_var']
        learningRate = self.params['learning_rate']
        I_Phi = self.params['I_Phi']
        I_Omega = self.params['I_Omega']
        miRNA = self.params['X']
        mRNA = self.params['Y']
        nData = self.params['n_sample']
        
        
        X = tf.placeholder(tf.float32, [None, M])
        Y = tf.placeholder(tf.float32, [None, N])

        U = tf.Variable(tf.random_uniform([M, K], 0.0, 1.0))
        V = tf.Variable(tf.random_uniform([N, K], 0.0, 1.0))
        mu = tf.Variable(tf.random_uniform([1, N], 0.0, 1.0))
        
        UV_T = tf.matmul(U, tf.transpose(V))
        # print UV_T.get_shape()
        VV_T = tf.matmul(V, tf.transpose(V))
        Y_hat = mu - tf.matmul(X, UV_T)
        
        
        cost1 = tf.reduce_mean( tf.div(tf.reduce_mean(tf.square(Y - Y_hat), 0), mRNA_var) )    
        cost2 = tf.reduce_sum(-tf.log( self.prob_I_Phi(UV_T, I_Phi, alpha) ) )
        cost3 = tf.reduce_sum(-tf.log( self.prob_I_Omega(VV_T, I_Omega, beta, N) ) )
        cost = cost1 + cost2 + cost3
        
        penalty1 = tf.reduce_mean( tf.maximum( tf.reduce_sum ( tf.abs(U), 0 ) - C1, 0) )
        penalty2 = tf.reduce_mean( tf.maximum( tf.reduce_sum ( tf.abs(V), 0 ) - C2, 0) )
        penalty3 = -tf.minimum( tf.reduce_min(U),0)
        # print penalty3.get_shape()    
        penalty4 = -tf.minimum( tf.reduce_min(V),0)
        penalty5 = -tf.minimum( tf.reduce_min(mu),0)
        penalty =  penalty1 + penalty2 + penalty3 + penalty4 + penalty5
        
        total_cost = cost + lambda1*penalty
        
        optimizer = tf.train.AdamOptimizer(learningRate)
        # optimizer = tf.train.GradientDescentOptimizer(learningRate)
        train = optimizer.minimize(total_cost)    
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)    

            n_batch = self.params['n_batch']
            n_iter = int(self.params['n_sample']/float(n_batch))
            for e in range(self.params['n_epoch']):
                for i in range(n_iter):                
                    print('iteration: {}'.format(i))
                
                    xs = miRNA[n_batch*i:n_batch*(i+1), :]
                    ys = mRNA[n_batch*i:n_batch*(i+1), :]

                    _, total_cost_val, c1, c2, c3, p1,p2,p3,p4,p5 = sess.run(
                        [train, total_cost, cost1, cost2, cost3, penalty1, penalty2, penalty3, penalty4, penalty5],
                        feed_dict={X: xs, Y: ys}
                    )
                    
                    #if i % 200 == 0:
                    #    info = "epoch = %d, batch = %d, total_cost = %.3f, cost = (%.3f,%.3f,%.3f), penalty = (%.3f,%.3f,%.3f,%.3f,%.3f)" % (
                    #        e, i, total_cost_val, c1,c2,c3, p1,p2,p3,p4,p5
                    #    )
                    #    # print info
                  

            q = UV_T.eval()
            w = U.eval()
            e = V.eval()
            
        return q,w,e
