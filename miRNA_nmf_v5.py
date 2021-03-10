import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle
import utils as u

import config

summary_flag = False

  
#-----------------------------------------------------------------------------------------------------    
def vnet(Omega, N, K, num, threshold=0.95, initial=None):

    # VT = common_nn(Omega, N*N, K, N, "vnet_model")
    # V = tf.transpose(VT)
    # Omega_hat = tf.matmul(V, VT)    
    V = u.weight_variable((N,K), name='V', initial=None)
    VT = tf.transpose(V)
    Omega_hat = tf.matmul(V, VT)    
    clip_V = V.assign(tf.maximum(0., V))    
    
    with tf.variable_scope("vnet_cost"):        
        val, id = tf.nn.top_k(VT, num)
        dif = val[:,0] - val[:,num-1]
        penalty = tf.reduce_mean(
            # In a row of V, make the diffence between the largest and the second largest > threshold
            tf.maximum(-(dif - threshold), 0)
        )                  
        # cost = tf.reduce_mean(
            # -Omega*tf.log(tf.sigmoid(Omega_hat))
        # )          
        cost = tf.reduce_mean(tf.square(Omega-Omega_hat)) 
    return V, cost, penalty, VT, clip_V
#-----------------------------------------------------------------------------------------------------    
def unet(VT, Phi, M, N, K, num, threshold=0.95, initial=None):
    # PhiT = tf.transpose(Phi)
    # UT = common_nn(PhiT, M*N, K, M, "unet_model")
    # PhiT_hat = tf.matmul(V, UT)
    
    # Phi_hat = tf.transpose(PhiT_hat)
    # U = tf.transpose(UT)

    U = u.weight_variable((M,K), name='U', initial=None)
    Phi_hat = tf.matmul(U, VT)    
    clip_U = U.assign(tf.maximum(0., U))    
    
    UT = tf.transpose(U)
    with tf.variable_scope("unet_cost"):        
        val, id = tf.nn.top_k(UT, num)
        dif = val[:,0] - val[:,num-1]
        penalty = tf.reduce_mean(
            tf.maximum(-(dif - threshold), 0)
        )                      

        # cost = tf.reduce_mean(
            # -Phi*tf.log(tf.sigmoid(Phi_hat)) -(1-Phi)*tf.log(tf.sigmoid(2-Phi_hat))
        # )   
        cost = tf.reduce_mean(tf.square(Phi-Phi_hat))         

    return U, cost, penalty, clip_U
#-----------------------------------------------------------------------------------------------------  
def wnet(U, VT, M, N, miRNA_vec, mRNA_vec, Sigma, Mu):   

    with tf.variable_scope("wnet_model"):      
        wb = u.bias_variable((1,), name='wb', initial=3.0)
        S = tf.sigmoid(2*wb*tf.matmul(U, VT)-wb)
        # S = tf.where(S>0.98, x=tf.ones(shape=tf.shape(S)), y=tf.zeros(shape=tf.shape(S)))
        # S = tf.matmul(U, VT)        
        # S = tf.where(S>wb, x=tf.ones(shape=tf.shape(S)), y=tf.zeros(shape=tf.shape(S)))
        # clip = lambda a: 0.5*tf.tanh(10*(a-0.5))+0.5
        # S = clip(S)
        
        W = u.weight_variable((M,N), name='W')
        # Mu = u.bias_variable((N,), name='Mu')
        WS = W * S
        y = tf.matmul(miRNA_vec, WS) + Mu
        
    with tf.variable_scope("wnet_cost"):    
        # penalty = 100*tf.maximum(-wb,0)

        penalty = tf.reduce_mean(
            tf.abs(W)
        ) #+ 100*tf.maximum(-wb,0)
        # penalty = tf.maximum(tf.reduce_max(tf.abs(W))-0.1,0)

        cost = tf.reduce_mean(tf.square(y-mRNA_vec)/Sigma)
        
        
    return W, S, WS, cost, penalty, wb
#-----------------------------------------------------------------------------------------------------  
def train(params):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model_file = config.Config.model_save
    summary_dir = 'summary_tiresias/'
    #ret_filename = config.Config.results + '.ret'    
    tf.reset_default_graph()
    
    M = params['M']
    N = params['N']
    K = params['K']

    lambda1 = params['lambda1']
    lambda2 = params['lambda2']
    lambda3 = params['lambda3']

    x = tf.placeholder(tf.float32, [None,M], name='x')    
    y = tf.placeholder(tf.float32, [None,N], name='y')
    Phi = tf.placeholder(tf.float32, [M,N], name='Phi')
    Omega = tf.placeholder(tf.float32, [N,N], name='Omega')
    Sigma = tf.placeholder(tf.float32, [N,], name='Sigma')
    Mu = tf.placeholder(tf.float32, [N,], name='Mu')

    V, vc, vp, VT, clip_V = vnet(Omega, N, K, params['num_v'], threshold=params['T'], initial=params['V_initial'])
    U, uc, up, clip_U = unet(VT, Phi, M, N, K, params['num_u'], threshold=params['T'], initial=params['U_initial'])
    W, S, WS, wc, wp, wb = wnet(U, VT, M, N, x, y, Sigma, Mu)
    clip = tf.group(clip_U, clip_V)

    total_cost = wc + lambda1*uc + lambda2*vc + lambda3*wp + lambda3*vp + lambda3*up


    tf.add_to_collection('tiresias', x)
    tf.add_to_collection('tiresias', y)
    tf.add_to_collection('tiresias', Phi)
    tf.add_to_collection('tiresias', Omega)
    tf.add_to_collection('tiresias', U)
    tf.add_to_collection('tiresias', V)
    tf.add_to_collection('tiresias', W)
    
    u.summary(total_cost, summary_flag)
    train_op = tf.train.AdamOptimizer(params['learning_rate']).minimize(total_cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()    
    sess.run(init)
    saver = tf.train.Saver()   
    
    if summary_flag:
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)    
        
    n_batch = params['n_batch']
    n_iteration = params['n_sample']/n_batch
    n_outer = 1 # do not change this
    _WS = 0
    for o in range(n_outer):
        if o == 0:
            Phi_in = params['I_Phi']
            #print np.sum(Phi_in)
            # temp = raw_input("enter:")
        else:
            m = np.max(np.abs(_WS))
            Phi_in = params['I_Phi'] * np.where(np.abs(_WS)>m/100., 1, 0)
            #print np.sum(Phi_in)
            # temp = raw_input("enter:")            
        
        
        for epoch in range(params['n_epoch']):
            for i in range(int(n_iteration)):
                x_batch = params['X'][(n_batch*i):(n_batch*(i+1)),:]
                y_batch = params['Y'][(n_batch*i):(n_batch*(i+1)),:]    
                
                _, _cost, _WS, _U, _V, _W, _S, _wc, _wp, _uc, _up, _vc, _vp, _wb = sess.run(
                    [train_op, total_cost, WS, U, V, W, S, wc, wp, uc, up, vc, vp, wb],
                    feed_dict={x: x_batch, y: y_batch, Phi: Phi_in, Omega: params['I_Omega'], Sigma: params['y_var'], Mu: params['y_mean']}
                )
                sess.run(clip)
                
                #if i % 1 == 0:
                #    print "epoch=%d step=%4d cost=%.3f (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)(%.2f)" %(epoch, i, _cost, _wc, _wp, _uc, _up, _vc, _vp, _wb)

                if summary_flag and (i % 100 == 0):
                    _merged = sess.run(merged, {x: x_batch, y: y_batch, Phi: params['I_Phi'], Omega: params['I_Omega'], Sigma: params['y_var'], Mu: params['y_mean']})
                    summary_writer.add_summary(_merged, n_iteration*epoch+i)    
                
    #if config.Config.save_enable == "Enable":                          
    #    saver.save(sess, model_file)
    #    with open(ret_filename, 'wb') as fdat:
    #        pickle.dump((_U, _V, _W), fdat)            

    return _S,_W,_U,_V
