import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

import config
prefix = config.Config.results
#----------------------------------------------------------------------------------------------------- 
def plot_heatmap(params, SW,U,V, S = None, corr=None, gt=None, putative=None, vmin = 0, vmax=1, prefix_aux=""):
    global prefix
    prefix = prefix + prefix_aux
    y = range(params['M'])
    x = range(params['N'])

    plt.figure()
    if not (params['gt'][2] is None):
        nc_mi = []
        nc_m = []
        for i in range(params['M']):
            for j in range(params['N']):
                if params['gt'][2][i,j] == 1:
                    nc_mi.append(i)
                    nc_m.append(j)
        plt.scatter(nc_m, nc_mi, s=15, marker='o', c='w', alpha=0.3, edgecolors = 'black')     
        
    if not (params['gt'][0] == [] or params['gt'][1] == []):
        for i in range(len(params['gt'][1])):
            c = params['gt'][3][params['gt'][0][i],params['gt'][1][i]]
            if c>0:
                color = 'k'
                scale = c/0.1
                if scale < 1:
                    scale = 0.5
                    color = 'w'
                scale = 0.5
                plt.scatter(params['gt'][1][i], params['gt'][0][i], s=scale*20.0, marker='^', c=color)    
            else:
                color = 'k'
                scale = -c/0.1
                if scale < 1:
                    scale = 0.5            
                    color = 'w'
                scale = 0.5                    
                plt.scatter(params['gt'][1][i], params['gt'][0][i], s=scale*20.0, marker='v', c=color)    

    # plt.imshow(regulationEst, interpolation='none',cmap=plt.get_cmap('gray'))    
    plt.imshow(SW, interpolation='none', cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xlabel('mRNA', fontsize=20)
    plt.ylabel('miRNA', fontsize=20)
    # plt.xticks(x, params['mRNA_names'], rotation='vertical')
    # plt.yticks(y, params['miRNA_names'])
    # plt.subplots_adjust(bottom=0.25)
    plt.savefig(prefix + 'sw_est.png')    

    if S is not None:
        plt.figure()
        if not (params['gt'][2] is None):
            nc_mi = []
            nc_m = []
            for i in range(params['M']):
                for j in range(params['N']):
                    if params['gt'][2][i,j] == 1:
                        nc_mi.append(i)
                        nc_m.append(j)
            plt.scatter(nc_m, nc_mi, s=15, marker='o', c='g', alpha=0.3, edgecolors = 'black')     
            
        if not (params['gt'][0] == [] or params['gt'][1] == []):
            for i in range(len(params['gt'][1])):
                c = params['gt'][3][params['gt'][0][i],params['gt'][1][i]]
                if c>0:
                    color = 'k'
                    scale = c/0.1
                    if scale < 1:
                        scale = 0.5
                        color = 'w'
                    scale = 0.5                    
                    plt.scatter(params['gt'][1][i], params['gt'][0][i], s=scale*20.0, marker='^', c=color)    
                else:
                    color = 'k'
                    scale = -c/0.1
                    if scale < 1:
                        scale = 0.5            
                        color = 'w'
                    scale = 0.5                    
                    plt.scatter(params['gt'][1][i], params['gt'][0][i], s=scale*20.0, marker='v', c=color)         
        plt.imshow(S, vmin=vmin, vmax=1, interpolation='none', cmap=plt.get_cmap('jet'))
        plt.colorbar()
        plt.xlabel('mRNA', fontsize=20)
        plt.ylabel('miRNA', fontsize=20)
        # plt.xticks(x, params['mRNA_names'], rotation='vertical')
        # plt.yticks(y, params['miRNA_names'])
        # plt.subplots_adjust(bottom=0.25)
        plt.savefig(prefix + 's_est.png')    

    if corr is not None:
        plt.figure()
        plt.imshow(corr, interpolation='none', cmap=plt.get_cmap('jet'))
        plt.colorbar()
        plt.xlabel('mRNA', fontsize=20)
        plt.ylabel('miRNA', fontsize=20)
        # plt.xticks(x, params['mRNA_names'], rotation='vertical')
        # plt.yticks(y, params['miRNA_names'])
        # plt.subplots_adjust(bottom=0.25)
        plt.savefig(prefix + 'corr.png')    

    if gt is not None:
        plt.figure()
        plt.imshow(gt, interpolation='none', cmap=plt.get_cmap('jet'))
        plt.colorbar()
        plt.xlabel('mRNA', fontsize=20)
        plt.ylabel('miRNA', fontsize=20)
        # plt.xticks(x, params['mRNA_names'], rotation='vertical')
        # plt.yticks(y, params['miRNA_names'])
        # plt.subplots_adjust(bottom=0.25)
        plt.savefig(prefix + 'gt.png')          

    if putative is not None:
        plt.figure()
        plt.imshow(putative, interpolation='none', cmap=plt.get_cmap('jet'))
        plt.colorbar()
        plt.xlabel('mRNA', fontsize=20)
        plt.ylabel('miRNA', fontsize=20)
        # plt.xticks(x, params['mRNA_names'], rotation='vertical')
        # plt.yticks(y, params['miRNA_names'])
        # plt.subplots_adjust(bottom=0.25)
        plt.savefig(prefix + 'putative.png')   
        
    plt.figure()
    plt.imshow(U, vmin=vmin, vmax=vmax, interpolation='none', cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xticks(np.arange(0,params['K'],5), np.arange(0,params['K'],5))
    # plt.xticks(range(params['K']), range(1,params['K']+1))
    # plt.yticks(y, params['miRNA_names'])
    plt.ylabel('miRNA', fontsize=20)
    plt.xlabel('module', fontsize=20)        
    # plt.subplots_adjust(bottom=0.20)
    plt.savefig(prefix + 'u_est.png')   

    plt.figure()
    plt.imshow(V, vmin=0, vmax=vmax, interpolation='none', cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.xticks(np.arange(0,params['K'],5), np.arange(0,params['K'],5))
    # plt.xticks(range(params['K']), range(1,params['K']+1))
    # plt.yticks(y, params['mRNA_names'])
    plt.ylabel('mRNA', fontsize=20)
    plt.xlabel('module', fontsize=20)                
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(prefix + 'v_est.png')        

    # plt.show()
#----------------------------------------------------------------------------------------------------- 
