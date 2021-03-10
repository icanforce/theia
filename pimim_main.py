import config
config.Config.select_menu('pimim_')
#---------------------------------------------------------------------
import numpy as np
import os
import pickle
#import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------
import miRNA_synthetic_data as syn
import parse_miRTarBase as gt
import parse_TargetScan as putative
import parse_BioGRID as ppi
import parse_gene as gene
import miRNA_load_expression as data_exp
import pimim_v2 as infer
#import miRNA_draw as draw
import miRNA_score as score
import miRNA_module as module
#-----------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------
def getStat(mRNA, mRNA_num):
    mean = np.zeros(mRNA_num)
    variance = np.zeros(mRNA_num)
    for i in range(mRNA_num):    
        mean[i] = np.mean(mRNA[:,i])
        variance[i] = np.var(mRNA[:,i])
    return mean, variance
#-----------------------------------------------------------------------------------------------------
def normalize(miRNA, mRNA, nInput, nOutput):
    mean, var = getStat(miRNA, nInput)
    # print mean
    # print var
    miRNA = np.divide(miRNA-mean, np.sqrt(var)) + 3*np.ones((1,1))
    mean, var = getStat(miRNA, nInput)
    # print mean
    # print var    
    mRNA_mean, mRNA_var = getStat(mRNA, nOutput)
    # print mRNA_mean
    # print mRNA_var    
    mRNA = np.divide(mRNA, mRNA_mean) + 10*np.ones((1,1))
    mRNA_mean, mRNA_var = getStat(mRNA, nOutput)
    # print mRNA_mean
    # print mRNA_var
    
    return miRNA, mRNA, mRNA_mean, mRNA_var
#-----------------------------------------------------------------------------------------------------    


if __name__ == '__main__':

    dataFile = config.Config.all_data_save
    
    miRNA, mRNA = [], []
    if not os.path.isfile(dataFile): 
        if config.Config.key == 'SYNTHETIC':
            pass
        else:
            gene_dic, _ = gene.parse_gene()
            print("loading expressions")
            d = data_exp.ExpData(gene_dic)
            miRNA_list, mRNA_list, miRNA, mRNA = d.getAllExp(mi_th=0.1, gene_th=0.1)
            # mRNA_list = mRNA_list[:10]
            # mRNA = mRNA[:,:10]
            nInput = len(miRNA_list)
            nOutput = len(mRNA_list)
            nData = len(miRNA)                
            corr = d.calcCorr()

            
            print("loading a putative matrix")
            putative = putative.parse_prediction(miRNA_list, mRNA_list)

            
            print("loading a ppi matrix")
            ppi = ppi.get_interaction(mRNA_list)
        

            print("loading ground-truth pairs")
            gt, gt_mi, gt_m = gt.parse_groundTruth(miRNA_list, mRNA_list)   
            
            # correct false negatives in putative interactions from experimentally-validated results
            putative = np.logical_or(putative,gt)*1
            
            aux = [[],[]]

        if config.Config.save_enable == "Enable":
            with open(dataFile, 'wb') as fdat:
                pickle.dump(
                    (
                        miRNA, mRNA, nInput, nOutput, miRNA_list, mRNA_list,
                        nData, gt, gt_mi, gt_m, putative, corr, ppi
                    ),
                    fdat
                )  
    else:
        miRNA, mRNA, nInput, nOutput, miRNA_list, mRNA_list, nData, gt, gt_mi, gt_m, putative, corr, ppi = pickle.load( open( dataFile, "rb" ) )


    miRNA, mRNA, mRNA_mean, mRNA_var = normalize(miRNA, mRNA, nInput, nOutput)        
    
    # print putative
    # print ppi
    # print gt    

    
    # print mRNA
    # print nInput, nOutput
   
    
    params = {
        'summary_active': False,
        'learning_rate': 0.1,
        'M': nInput,
        'N': nOutput,
        'K': nInput // 5,
        'I_Phi': putative,
        'I_Omega': ppi,
        'X': miRNA,
        'Y': mRNA,
        'y_var': mRNA_var,
        'n_sample': nData,
        'n_batch': 1000,        
        'n_epoch': 1000,
        'miRNA_names': miRNA_list,
        'mRNA_names': mRNA_list,
        'gt': [gt_mi, gt_m, putative, corr, gt]
    }    
    t = infer.Trainer(params)
    SW,U,V = t.train()    
    # print np.max(SW), np.max(U), np.max(V)
    # print SW[0:4,:]
    
    # draw.plot_heatmap(params, SW,np.absolute(U),V, vmin = None, vmax=None)

    np.save('OUTPUT_estimation.npy', SW)
    np.save('OUTPUT_U.npy', U)
    np.save('OUTPUT_V.npy', V)

    from miRNA_module2 import find_modules
    modules = find_modules(miRNA_list, mRNA_list, U, V, 0.5)
    from json import dump
    with open('OUTPUT_modules.json', 'w') as f:
        dump(modules, f)

    #corr_th = 0.0
    #gt_sign = ((corr*gt)<(-corr_th))*(-1) + ((corr*gt)>corr_th)*(1)
    #p, r, f, a, tp, fp, tn, fn = score.calcScore2(SW, gt_sign, 0, putative)
    #print(('Precision = {}\nRecall = {}\nF1-Score = {}\nAccuracy = {}\n' +
    #        'TP = {}\nFP = {}\nTN = {}\nFN = {}').format(p, r, f, a, tp, fp, tn, fn))
