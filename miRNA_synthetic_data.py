from miRNA_module2 import find_modules
import numpy as np
import random
import scipy.stats


def genData(nSample, M, N, K,
            num_modules_per_item,
            num_putative_per_mirna,
            signal_strength,
            variance_mirna=1,
            variance_mrna=1,
            seed=0):
    """
        Sythetic data generator

        Arg:
            signal_strength:    regulation weight (or strength of regulation)
            variance_mirna:         variance of an miRNA expression
            variance_mrna:          variance of an mRNA expression
        Return:
            x_list:     list of miRNA vectors, each of which is a numpy array of dim(1,M)
            y_list:     list of mRNA vectors, each of which is a numpy array of dim(1,N)
    """

    random.seed(seed)

    x_list = []
    y_list = []
    gt_mi = []
    gt_m = []
    x_name = [i for i in range(M)]
    y_name = [i for i in range(N)]
    
    U = np.zeros((M, K))
    V = np.zeros((N, K))
    #true_module_mi = []
    #true_module_m = []    
    
    module = range(K)
    for i in range(M):
        select_module = random.sample(module, random.randint(1,num_modules_per_item))
        U[i,select_module] = 1
        #true_module_mi.append(select_module)
    for i in range(N):
        select_module = random.sample(module, random.randint(1,num_modules_per_item))
        V[i,select_module] = 1
        #true_module_m.append(select_module)

    #true_module = list(zip(true_module_mi, true_module_m))
    true_module = find_modules(x_name, y_name, U, V, 0.99)

    ground_truth = (np.matmul(U, np.transpose(V))>0)*1
    ppi = (np.matmul(V, np.transpose(V))>0)*1
    for i in range(N):
        ppi[i,i] = 1

    prediction = np.copy(ground_truth)
    for i in range(M):    
        mask = list(np.where(ground_truth[i,:]==0)[0])
        num = num_putative_per_mirna - (N-len(mask))
        
        if num > 0:
            pred = random.sample(mask, num)
            prediction[i, pred] = 1
        
    ground_truth_sign = -1*np.copy(ground_truth)
    for i in range(M):    
        mask = list(np.where(ground_truth[i,:]>0)[0])
        sign = random.sample(mask, int(len(mask)*0.2) )
        ground_truth_sign[i, sign] = 1  
    
    for i in range(M):
        for j in range(N):
            if ground_truth[i,j]>0:
                gt_mi.append(i)
                gt_m.append(j)

    for s in range(nSample):
        x = np.zeros((1, M))
        for i in range(M):
            x[0,i] = np.random.normal(0, variance_mirna)
        x_list.append(x)
        
    x_list = np.vstack(x_list)        
    x_list_s = np.divide(x_list, np.sqrt(variance_mirna)) + 3*np.ones((1,1)) # N(3, 1)
    
    for s in range(nSample):        
        y = np.zeros((1, N))
        for j in range(N):
            gt = ground_truth[:, j]
            idx = [i for i in range(M) if gt[i] == 1.0]
            regulation = 0.0

            for g in idx:
                regulation += ground_truth_sign[g,j] * signal_strength * x_list_s[s,g]
            y[0,j] = np.random.normal(10 + regulation, np.sqrt(variance_mrna)) # N(10+reg, 1)
        
        y_list.append(y)    
        
    y_list = np.vstack(y_list)
    corr = np.zeros((M, N))
    for i in range(M):
        sep = ','
        for j in range(N):
            r, p = scipy.stats.pearsonr(x_list[:,i], y_list[:,j])
            corr[i, j] = r

    return x_list, y_list, x_name, y_name, ground_truth, gt_mi, gt_m, prediction, ground_truth_sign, ppi, corr, true_module 
 
