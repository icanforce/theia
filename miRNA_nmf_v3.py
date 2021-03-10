import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle
import utils as u

import config

summary_flag = False


# num_v = 3
# num_u = 3
# num_v = 6
# num_u = 4

# T = 0.95
    
#-----------------------------------------------------------------------------------------------------    
def common_nn(input, L, R, C, scope_name):

    with tf.variable_scope(scope_name):  
        x = tf.reshape(input, (1,L))
        
        # w1 = u.weight_variable((L,L), name='w1')
        # b1 = u.bias_variable((L,), name='b1')
        # h1 = u.leaky_relu(tf.matmul(x, w1) + b1, leak = 0.2)

        w2 = u.weight_variable((L,R*C), name='w2')
        #b2 = u.bias_variable((R*C,), name='b2')
        h2 = tf.sigmoid(tf.matmul(x, w2))# + b2)
        
        Mat = tf.reshape(h2, (R, C))
        
    return Mat
#-----------------------------------------------------------------------------------------------------    
def vnet(Omega, N, K, num, threshold=0.95):

    V = common_nn(Omega, N*N, N, K, "vnet_model")
    VT = tf.transpose(V)
    Omega_hat = tf.matmul(V, VT)    
    vb = u.bias_variable((1,), name='vb')
    
    with tf.variable_scope("vnet_cost"):        
        val, id = tf.nn.top_k(V, num)
        dif = val[:,0] - val[:,num-1]
        penalty = tf.reduce_mean(
            # In a row of V, make the diffence between the largest and the second largest > threshold
            tf.maximum(-(dif - threshold), 0)
        )                  
        # cost = tf.reduce_mean(tf.square(Omega*vb-Omega_hat))    
        cost = tf.reduce_mean(
            -Omega*tf.log(tf.sigmoid(Omega_hat))
        )          
    return V, cost, penalty, VT, vb
#-----------------------------------------------------------------------------------------------------    
def unet(VT, Phi, M, N, K, num, threshold=0.95):

    U = common_nn(Phi, M*N, M, K, "unet_model")
    Phi_hat = tf.matmul(U, VT)
    ub = u.bias_variable((1,), name='ub')
    
    with tf.variable_scope("unet_cost"):        
        val, id = tf.nn.top_k(U, num)
        dif = val[:,0] - val[:,num-1]
        penalty = tf.reduce_mean(
            tf.maximum(-(dif - threshold), 0)
        )                      
        # cost = tf.reduce_mean(tf.square(Phi*ub-Phi_hat))    
        cost = tf.reduce_mean(
            -Phi*tf.log(tf.sigmoid(Phi_hat)) -(1-Phi)*tf.log(tf.sigmoid(2-Phi_hat))
        )        

    return U, cost, penalty, ub
#-----------------------------------------------------------------------------------------------------  
def wnet(U, VT, M, N, miRNA_vec, mRNA_vec, Sigma, Mu):   

    with tf.variable_scope("wnet_model"):      
        wb = u.bias_variable((1,), name='wb', initial=3.0)
        S = tf.sigmoid(2*1*tf.matmul(U, VT)-1)
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
    # model_file = config.Config.model_save
    # summary_dir = 'summary_tiresias/'
    # ret_filename = config.Config.results + '.ret'    
    
    M = params['M']
    N = params['N']
    K = params['K']

    x = tf.placeholder(tf.float32, [None,M], name='x')    
    y = tf.placeholder(tf.float32, [None,N], name='y')
    Phi = tf.placeholder(tf.float32, [M,N], name='Phi')
    Omega = tf.placeholder(tf.float32, [N,N], name='Omega')
    Sigma = tf.placeholder(tf.float32, [N,], name='Sigma')
    Mu = tf.placeholder(tf.float32, [N,], name='Mu')

    V, vc, vp, VT, vb = vnet(Omega, N, K, params['num_v'], threshold=params['T'])
    U, uc, up, ub = unet(VT, Phi, M, N, K, params['num_u'], threshold=params['T'])
    W, S, WS, wc, wp, wb = wnet(U, VT, M, N, x, y, Sigma, Mu)

    total_cost = wc + 0.5*uc + 0.5*vc + 0.250*wp + 0.25*vp + 0.25*up


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
    tf.reset_default_graph()
    
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
            # print(np.sum(Phi_in))
            # temp = raw_input("enter:")
        else:
            m = np.max(np.abs(_WS))
            Phi_in = params['I_Phi'] * np.where(np.abs(_WS)>m/100., 1, 0)
            # print(np.sum(Phi_in))
            # temp = raw_input("enter:")            
        
        
        for epoch in range(params['n_epoch']):
            # con = np.concatenate((params['X'], params['Y']), axis=1)
            # np.random.shuffle(con)
            # params['X'], params['Y'] = np.split(con, [M], axis=1)
            for i in range(int(n_iteration)):
                x_batch = params['X'][(n_batch*i):(n_batch*(i+1)),:]
                y_batch = params['Y'][(n_batch*i):(n_batch*(i+1)),:]    
                
                _, _cost, _WS, _U, _V, _W, _S, _wc, _wp, _uc, _up, _vc, _vp, _vb, _ub, _wb = sess.run(
                    [train_op, total_cost, WS, U, V, W, S, wc, wp, uc, up, vc, vp, vb, ub, wb],
                    feed_dict={x: x_batch, y: y_batch, Phi: Phi_in, Omega: params['I_Omega'], Sigma: params['y_var'], Mu: params['y_mean']}
                )
                
                #if i % 1 == 0:
                #    print("epoch=%d step=%4d cost=%.3f (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)(%.2f,%.2f,%.2f)" %(epoch, i, _cost, _wc, _wp, _uc, _up, _vc, _vp, _vb, _ub, _wb))
                    
                if summary_flag and (i % 100 == 0):
                    _merged = sess.run(merged, {x: x_batch, y: y_batch, Phi: params['I_Phi'], Omega: params['I_Omega'], Sigma: params['y_var'], Mu: params['y_mean']})
                    summary_writer.add_summary(_merged, n_iteration*epoch+i)    
                
    #if config.Config.save_enable == "Enable":                          
    #    saver.save(sess, model_file)
    #    with open(ret_filename, 'wb') as fdat:
    #        pickle.dump((_U, _V, _W), fdat)            

    return _S,_W,_U,_V
    
