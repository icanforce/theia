import os
import random
import re
import pickle
import numpy as np

import config
    
#-----------------------------------------------------------------------------------------------------  
class Trainer:
    def __init__(self, params):
        
        self.X = [0,0]
        self.X[0] = np.concatenate((np.maximum(params['X'],0),np.maximum(-params['X'],0)), axis=1)
        self.X[1] = np.concatenate((np.maximum(params['Y'],0),np.maximum(-params['Y'],0)), axis=1)
        self.S = params['n_sample']
        self.M = 2*params['M']
        self.N = 2*params['N']
        self.K = params['K']

        self.A = np.tile(params['I_Omega'], (2,2))
        self.B = np.tile(params['I_Phi'], (2,2))
        
        self.iteration = params['n_iteration']
        self.th = params['threshold']
        
        self.H = [0,0]
        self.H[0] = np.random.normal(0,.1,(self.K, self.M))
        self.H[1] = np.random.normal(0,.1,(self.K, self.N))
        self.W = np.random.normal(0,.1,(self.S, self.K))
        
        self.lambda1 = params['lambda1']#0.001
        self.lambda2 = params['lambda2']#0.001
        self.gamma1 = params['gamma1']#0.15
        self.gamma2 = params['gamma2']#0.15

        #self.model_filename = config.Config.model_save + 'SNMNMF.model'
        #self.ret_filename = config.Config.results + 'SNMNMF.ret'
#-----------------------------------------------------------------------------------------------------      
    def norm(self, W=None, H1=None, H2=None):
        ret = 0
        if W is None:
            W=self.W
        if H1 is None:
            H1=self.H[0]
        if H2 is None:
            H2=self.H[1]
        
        H = [H1, H2]        
        for i in [0,1]:
            ret += np.linalg.norm(self.X[i]- np.matmul(W, H[i]))
        return ret
#-----------------------------------------------------------------------------------------------------              
    def step2(self):
        nu = 0
        de = 0
        for i in [0,1]:
            nu += np.matmul(self.X[i], np.transpose(self.H[i]))
            de += np.matmul(np.matmul(self.W, self.H[i]), np.transpose(self.H[i]))
        de += self.gamma1/2.*self.W
        
        # print np.shape(nu), np.shape(de)
        
        self.W = self.W * nu / (de + np.finfo(float).eps)
#-----------------------------------------------------------------------------------------------------              
    def step3(self):
        nu1 = np.matmul(np.transpose(self.W), self.X[0])
        nu1 += self.lambda2/2.*np.matmul(self.H[1], np.transpose(self.B))
        de1 = np.matmul(np.transpose(self.W), self.W)+self.gamma2*np.ones((self.K,self.K))
        de1 = np.matmul(de1, self.H[0])
        # print np.shape(nu1), np.shape(de1)
        
        nu2 = np.matmul(np.transpose(self.W), self.X[1])
        nu2 += self.lambda1*np.matmul(self.H[1], self.A)
        nu2 += self.lambda2/2.*np.matmul(self.H[0], self.B)
        de2 = np.matmul(np.transpose(self.W), self.W)+self.gamma2*np.ones((self.K,self.K))
        de2 = np.matmul(de2, self.H[1])
        # print np.shape(nu2), np.shape(de2)
        
        self.H[0] = self.H[0] * nu1 / (de1 + np.finfo(float).eps)        
        self.H[1] = self.H[1] * nu2 / (de2 + np.finfo(float).eps)   
#-----------------------------------------------------------------------------------------------------                  
    def comodule(self):        
        TH = [0,0]     
        comodule = [np.zeros(np.shape(self.H[0])), np.zeros(np.shape(self.H[1]))]
        for i in [0,1]:
            m = np.mean(self.H[i], axis=1)
            s = np.std(self.H[i], axis=1)
            TH[i] = m + self.th * s
        for k in range(self.K):
            comodule[0][k,:] = self.H[0][k,:] > TH[0][k]
            comodule[1][k,:] = self.H[1][k,:] > TH[1][k]
        # print self.H[0]
        # print self.H[1]
        # print self.W
        
        U = 1.0 * np.transpose(np.logical_or(comodule[0][:,:int(self.M/2)], comodule[0][:,int(self.M/2):]))
        V = 1.0 * np.transpose(np.logical_or(comodule[1][:,:int(self.N/2)], comodule[1][:,int(self.N/2):]))
        
        return U, V
#-----------------------------------------------------------------------------------------------------                  
    def train(self):
        for t in range(self.iteration):
            self.step2()
            self.step3()
            # print t, self.norm()
                    
        #if config.Config.save_enable == "Enable":                  
        #    with open(self.model_filename, 'wb') as f:
        #        pickle.dump([self.W, self.H[0], self.H[1]], f)
                
        U, V = self.comodule()
        S = np.matmul(U, np.transpose(V))
        return S, U, V
#-----------------------------------------------------------------------------------------------------          
