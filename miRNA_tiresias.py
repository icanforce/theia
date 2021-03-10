import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
# import config
#-----------------------------------------------------------------------------------------------------
nStep = 1
nData = 20000
nBatch = 1
    
nInput = 10
nOutput = 10

miRNA = []
mRNA = []

lambda2 = 0.2
lambda3 = 2.0

mRNA_stat = None
miRNA_name = []
mRNA_name = []
gt = None
linear = True

featureRatio = 38.0
nFeature = 1
#-----------------------------------------------------------------------------------------------------
def dropout(nodes, keep_prob):
    nodes_drop = tf.nn.dropout(nodes, keep_prob)
    return nodes_drop
#-----------------------------------------------------------------------------------------------------
def encode(x, y):
    """
        Extract features by an autoencoder
        
        Arg:
            x:  miRNA, tensor of dim(None, nInput)
            y:  mRNA, tensor of dim(None, nOutput)
        Return:
            feat:  feature values, tensor of dim(None, int((nInput+nOutput)/featureRatio))
    """
    # print (nOutput+nInput), (nOutput+nInput)/featureRatio, featureRatio
    # assert int((nOutput+nInput)/featureRatio) >= 1
    
    x_y = tf.concat(axis=1, values=[x, y])
    q0 = tf.Variable(tf.random_uniform([nOutput+nInput, nFeature], -1.0, 1.0))
    b0 = tf.Variable(tf.random_uniform([1, nFeature], -1.0, 1.0))
    feat = tf.sigmoid(tf.matmul(x_y, q0) + b0)
    return feat, q0, b0
#-----------------------------------------------------------------------------------------------------
def encode_fix(x, y, q, b):
    """
        Extract features by an autoencoder
        
        Arg:
            x:  miRNA, tensor of dim(None, nInput)
            y:  mRNA, tensor of dim(None, nOutput)
            q:  weight, constant numpy array of dim(nOutput+nInput, int((nOutput+nInput)/featureRatio))
            b:  bias, constant numpy array of dim(1, int((nOutput+nInput)/featureRatio))
        Return:
            feat:  feature values, tensor of dim(None, int((nInput+nOutput)/featureRatio))
    """
    x_y = tf.concat(axis=1, values=[x, y])
    q0 = tf.Variable(q)
    b0 = tf.Variable(b)    
    feat = tf.sigmoid(tf.matmul(x_y, q0) + b0)
    return feat
#-----------------------------------------------------------------------------------------------------
def encoderTraining(x, y, feat, learningRate):
    x_y = tf.concat(axis=1, values=[x, y])
    q0 = tf.Variable(tf.random_uniform([nFeature, nOutput+nInput], -1.0, 1.0))
    b0 = tf.Variable(tf.random_uniform([1, nOutput+nInput], -1.0, 1.0))
    x_y_hat = tf.matmul(feat, q0) + b0
    
    loss = tf.reduce_mean(tf.square(x_y - x_y_hat))
    
    optimizer = tf.train.AdamOptimizer(learningRate)
    # optimizer = tf.train.GradientDescentOptimizer(learningRate)
    train = optimizer.minimize(loss)
    return train, loss
#-----------------------------------------------------------------------------------------------------    
def g(feat):
    """
        Estimate of P(s|x,y) or equivalently E(s|x,y)
        
        Arg:
            feat:  feature values, tensor of dim(None, int((nInput+nOutput)/featureRatio))
        Return:
            s:  membership matrix, tensor of dim(None, nInput*nOutput)
    """
    const = 1.0
    q = tf.Variable(tf.random_uniform([nFeature, nOutput*nInput], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([1, nOutput*nInput], -1.0, 1.0))
    s_in = tf.matmul(feat, q) + b
    s = tf.sigmoid( const * s_in )    
    
    clip = lambda a: 0.5*tf.tanh(10*(a-0.5))+0.5

    return clip(s)
#-----------------------------------------------------------------------------------------------------
def linearRegulation(x, s, *w):
    """
        Linear regulation function
        
        Arg:
            x:      miRNA, tensor of dim(None, nInput)
            s:      membership matrix, tensor of dim(None, nInput*nOutput)
            w:      weight matrix, tuple of a single tensor of dim(nInput, nOutput)
        Return:
            reg:   magnitude of regulation, tensor of dim(None, nOutput)
    """

    cond = tf.reshape(tf.cast(tf.constant(gt[2]), tf.float32), [1, nOutput*nInput])    
    ws_in = tf.multiply(tf.multiply(s, cond), tf.reshape(w[0], [1, nOutput*nInput]))
    # print ws_in.get_shape()
    ws = tf.reshape(ws_in, [-1, nInput, nOutput])    
    # What is being done below is matrix(x[None, 1, nInput]) * matrix(ws[None, nInput, nOutput])
    reg_in = tf.matmul(tf.reshape(x, [-1, 1, nInput]), ws)
    reg = tf.reshape(reg_in, [-1, nOutput])
    # print down.get_shape()
    return reg, [None]
#-----------------------------------------------------------------------------------------------------
def nonlinearRegulation(x, s, *w):
    """
        Multi-layer perceptron regulation function
        
        Arg:
            x:      miRNA, tensor of dim(None, nInput)
            s:      membership matrix, tensor of dim(None, nInput*nOutput)
            w:      tuple of weight matrices, tuple of tensors of two dimensions.
        Return:
            reg:   magnitude of regulation, tensor of dim(None, nOutput)
    """

    cond = tf.reshape(tf.cast(tf.constant(gt[2]), tf.float32), [1, nOutput*nInput])    
    ws_in = tf.multiply(tf.multiply(s, cond), tf.reshape(w[0], [1, nOutput*nInput]))    
    # ws_in = tf.mul(s, tf.reshape(w[0], [1, nOutput*nInput]))
    # print ws_in.get_shape()
    ws = tf.reshape(ws_in, [-1, nInput, nOutput])    
    b1 = tf.Variable(tf.random_uniform([1, nOutput], -1.0, 1.0))
    # What is being done below is matrix(x[None, 1, nInput]) * matrix(ws[None, nInput, nOutput])
    firstLayer_in = tf.matmul(tf.reshape(x, [-1, 1, nInput]), ws)
    firstLayer_in2 = tf.reshape(firstLayer_in, [-1, nOutput]) + b1
    firstLayer = tf.nn.relu( firstLayer_in2 )  
    
    b2 = tf.Variable(tf.random_uniform([1, nOutput], -1.0, 1.0))
    secondLayer_in = tf.matmul(firstLayer, w[1]) + b2
    secondLayer = tf.nn.relu( secondLayer_in )  
    
    reg = tf.matmul(secondLayer, w[2])
    
    return reg, [firstLayer, secondLayer]
#-----------------------------------------------------------------------------------------------------        
def f(x, s, m, *w):    
    """
        Estimate of E(y|x,s)
        
        Arg:
            x:      miRNA, tensor of dim(None, nInput)
            s:      membership matrix, tensor of dim(None, nInput*nOutput)
            m:      initial value for E(y|x,0), tensor of dim(None, nOutput)            
            w:      tuple of weight matrices of the regulation function, tuple of tensors of two dimensions.
        Return:
            mean:   E(y|x,s), tensor of dim(None, nOutput)
            m:      E(y|x,0), tensor of dim(None, nOutput)
    """

    if linear == True:
        regulation = linearRegulation
    else:
        regulation = nonlinearRegulation
    reg, aux = regulation(x, s, *w)

    mean = m + reg
    return mean, m, aux
#-----------------------------------------------------------------------------------------------------        
def fPre(x, s, m):    
    """
        xx
        
        Arg:
            x:  miRNA, tensor of dim(None, nInput)
            s:  membership matrix, tensor of dim(None, nInput*nOutput)
            m:      initial value for E(y|x,0), tensor of dim(None, nOutput)            
        Return:
            mean:   E(y|x,s), tensor of dim(None, nOutput)
            m:      E(y|x,0), tensor of dim(None, nOutput)
    """
    ws = tf.reshape(s, [-1, nInput, nOutput])    
    reg_in = tf.matmul(tf.reshape(x, [-1, 1, nInput]), ws)
    reg = tf.reshape(reg_in, [-1, nOutput])
    
    mean = m + reg
    return mean, m
#-----------------------------------------------------------------------------------------------------    
def costPre(s, condition):
    """
        Calculates the cost of the system
        
        Arg:
            s:      membership matrix, tensor of dim(None, nInput*nOutput)
            condition:  necessary conditions for inputs and outputs to be interacting
        Return:
            loss:   tensor of type float
    """
    cond = tf.reshape(tf.cast(tf.constant(condition), tf.float32), [1, nOutput*nInput])
    loss = tf.reduce_mean(tf.square(s - cond))

    return loss
#-----------------------------------------------------------------------------------------------------    
def cost(y, mean, m, s, rho, lam2, lam3):
    """
        Calculates the cost of the system
        
        Arg:
            y:      mRNA, tensor of dim(None, nOutput)
            mean:   E(y|x,s), tensor of dim(None, nOutput)
            m:      E(y|x,0), tensor of dim(None, nOutput)
            s:      membership matrix, tensor of dim(None, nInput*nOutput)
            rho:    specifies the sparcity of membership matrix s
            lam2:    Controls the sensitivity of the constraint m >= 0, type float
            lam3:    Controls the sensitivity of the constraint activeRatio == rho, type float
        Return:
            loss:   tensor of type float
    """
    # penalty1 = -tf.minimum( tf.reduce_min(w), 0)   
    # penalty2 = tf.reduce_mean( -tf.minimum( tf.reduce_min(m-mRNA_stat[0], 1),0) )   
    penalty2 = tf.reduce_mean( -tf.minimum( tf.reduce_min(m, 1),0) )   

    
    cond = tf.reshape(tf.cast(tf.constant(gt[2]), tf.float32), [1, nOutput*nInput])
    activeRatio = tf.reduce_sum(tf.multiply(s,cond), 1)/(nOutput*nInput)
    # activeRatio = tf.reduce_sum(s, 1)/(nOutput*nInput)
    kl = rho*tf.log(rho/activeRatio) + (1-rho)*tf.log( (1-rho)/(1-activeRatio) )
    penalty3 = tf.reduce_mean(kl)
    
    loglike = tf.reduce_mean( tf.div(tf.reduce_mean(tf.square(mean - y), 0), mRNA_stat[1]) )
    
    loss = loglike + lam2*penalty2 + lam3*penalty3
    return loss, loglike, penalty2, penalty3
#-----------------------------------------------------------------------------------------------------
def training(cost, learningRate):
    optimizer = tf.train.AdamOptimizer(learningRate)
    # optimizer = tf.train.GradientDescentOptimizer(learningRate)
    train = optimizer.minimize(cost)
    return train
#-----------------------------------------------------------------------------------------------------
def lastFewLoss(lossList, newLoss):
    """
        Maintains the last N loss values

        Arg:
            lossList:   list of loss values, list of length N
            newLoss:    float
        Return:
            average loss
    """

    # N = 500
    N = nData
    if len(lossList) == N:
        lossList.pop(0)
        lossList.append(newLoss)
    else:
        lossList.append(newLoss)
    return sum(lossList)/float(len(lossList))    
#-----------------------------------------------------------------------------------------------------    
def init(_nStep, _nData, _nBatch, _nInput, _nOutput, _miRNA, _mRNA, _mRNA_stat, _miRNA_name, _mRNA_name, _gt, _linear = True, _misc = None):
    global nStep, nData, nBatch, nInput, nOutput, miRNA, mRNA, mRNA_stat, miRNA_name, mRNA_name, gt, linear

    nStep = _nStep
    nData = _nData
    nBatch = _nBatch        
    nInput = _nInput
    nOutput = _nOutput
    miRNA = _miRNA
    mRNA = _mRNA
    mRNA_stat = _mRNA_stat
    miRNA_name, mRNA_name = _miRNA_name, _mRNA_name
    gt = _gt
    linear = _linear
    if _misc != None:
        global featureRatio, lambda2, lambda3
        featureRatio = _misc[0]
        lambda2 = _misc[1]
        lambda3 = _misc[2]
        
#-----------------------------------------------------------------------------------------------------    
 
    
#-----------------------------------------------------------------------------------------------------    
def runTraining(learningRate, rho, preTrain = True):
    """
        Main loop for training

        Arg:
            learningRate:   learning rate
            rho:            specifies the sparcity of membership matrix s
        Return:
            lossAve:            average of last few losses over training, type float
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
    learningRate = float(learningRate)
    
    x = tf.placeholder(tf.float32, [None, nInput])
    y = tf.placeholder(tf.float32, [None, nOutput])
    keep_prob = tf.placeholder(tf.float32)

    m = tf.Variable(tf.random_uniform([1, nOutput], 0.0, 1.0))
    if linear == True:
        w = tf.Variable(tf.random_uniform([nInput, nOutput], -1.0, 0.0))
    else:
        w1 = tf.Variable(tf.random_uniform([nInput, nOutput], -1.0, 1.0))
        w2 = tf.Variable(tf.random_uniform([nOutput, nOutput], -1.0, 1.0))
        w3 = tf.Variable(tf.random_uniform([nOutput, nOutput], -1.0, 1.0))

    feat0, q, b = encode(x, y)
    trainOpEnc, lossEnc = encoderTraining(x, y, feat0, learningRate/5.0)
        
    init0 = tf.global_variables_initializer()        

    encFile = 'encfile.dat'
    if not os.path.isfile(encFile): 
        with tf.Session() as sess:
            sess.run(init0)
            cnt = 0
            while cnt < 1000:
                for i in range(nData):
                    xs = miRNA[i:(i+1)]
                    ys = mRNA[i:(i+1)]

                    _, loss_val = sess.run([trainOpEnc, lossEnc], feed_dict={x: xs, y: ys})
                    
                    if i % 100 == 0:
                        info = "Encoder-training, cnt = %d, i = %d, cost = %.4f" % (cnt, i, loss_val)
                        print(info)
                    cnt += 1
                #if cnt>10000 and loss_val < 0.1:
                #    break
            q_fix = q.eval()
            b_fix = b.eval()
        
            if False:
                with open(encFile, 'wb') as fdat:
                    pickle.dump((q.eval(), b.eval()), fdat)  
    else:
        q, b = pickle.load( open( encFile, "rb" ) )
        q_fix = q
        b_fix = b
        
    feat = encode_fix(x, y, q_fix, b_fix)
    sHat_pre = g(feat)
    sHat = dropout(sHat_pre, keep_prob)
    
    # Pre-training to learn the initial weights of g(x,y)
    lossPre= costPre(sHat, gt[2])
    trainOpPre = training(lossPre, learningRate)
    
    # Main training
    if linear == True:
        mean, m, aux = f(x, sHat, m, w)
    else:
        mean, m, aux = f(x, sHat, m, w1, w2, w3)


    loss, loglike, penalty2, penalty3 = cost(y, mean, m, sHat, rho, lambda2, lambda3)
    trainOp = training(loss, learningRate)
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    modelFile = 'model_' + str(learningRate) + '_' + str(rho) + '_' + 'tiresias'
    estimationResult = None
    listOfLosses = []
    lossAve = 0
    with tf.Session() as sess:

        if 1:    
        # if not os.path.isfile(modelFile+'.meta'): 
            sess.run(init)
            
            # print "learningRate = %f, rho = %f" % (learningRate, rho)

            # Pre-training loop
            if preTrain == True:
                pre_cnt = 0
                while pre_cnt<10000:
                    
                    for i in range(int(nData/nBatch)):
                        xs = miRNA[(nBatch*i):(nBatch*(i+1)),:]
                        ys = mRNA[(nBatch*i):(nBatch*(i+1)),:]

                        _, loss_val = sess.run([trainOpPre, lossPre], feed_dict={x: xs, y: ys, keep_prob: 1.0})
                        pre_cnt += 1
                        
                        if i % 200 == 0:
                            info = "Pre-training, pre_cnt = %d, batch = %d, cost = %.4f" % (pre_cnt, i, loss_val)
                            # info = "Pre-training, batch = %d, cost = %.4f" % (i, loss_val)
                            print(info)
                        
            # Main training loop            
            for step in range(nStep):
                for i in range(int(nData/nBatch)):
                    # print i
                    xs = miRNA[(nBatch*i):(nBatch*(i+1)),:]
                    ys = mRNA[(nBatch*i):(nBatch*(i+1)),:]

                    _, loss_val, loglike_val, penalty2_val, penalty3_val = sess.run([trainOp, loss, loglike, penalty2, penalty3], feed_dict={x: xs, y: ys, keep_prob: 1.0})
                    # lossAve = lastFewLoss(listOfLosses, float(loss_val))
                    lossAve = lastFewLoss(listOfLosses, float(loglike_val))
                    
                    if i % 1000 == 0:
                    #if i == nData-1:
                        info = "step = %d, batch = %d, cost = %.4f" % (step, i, lossAve)
                        print(info)
                        print("loglike, penalty2, penalty3 = %.4f, %.4f, %.4f" % (loglike_val, penalty2_val, penalty3_val))
                        
            if False:
                save_path = saver.save(sess, modelFile)                  
        else:
            saver.restore(sess, modelFile)
            
            
        m_val = m.eval()
        #print
        #print m_val
        # print w.eval()

        print('**********')
        sAve = np.zeros((1, nInput*nOutput))
        for i in range(nData):
            s_val = sess.run(sHat, feed_dict={x: miRNA[i:(i+1)], y: mRNA[i:(i+1)], keep_prob: 1.0})
            sAve += s_val
        sAve = sAve/nData

        regulationEst = np.zeros((nInput, nOutput))
        xUnit = np.eye(nInput)
        sTemp = tf.convert_to_tensor(sAve, dtype=tf.float32)
        for i in range(nInput):
            xTemp = tf.convert_to_tensor(xUnit[i,:], dtype=tf.float32)
            if linear == True:
                reg_val, aux = linearRegulation(xTemp, sTemp, w)
            else:
                reg_val, aux = nonlinearRegulation(xTemp, sTemp, w1, w2, w3)
            regulationEst[i, :] = reg_val.eval()
            
        sAve = sAve.reshape((nInput, nOutput))
        np.set_printoptions(precision=3)   
        # print sAve
    # print regulationEst
    
    #plot_heatmap(learningRate, rho, regulationEst)
   
            
    return -1*lossAve, regulationEst

