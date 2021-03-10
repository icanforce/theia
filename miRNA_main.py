import config
config.Config.select_menu()
#---------------------------------------------------------------------
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------
import miRNA_synthetic_data as syn
import parse_miRTarBase as gt
import parse_TargetScan as putative
import parse_BioGRID as ppi
import parse_gene as gene
import miRNA_load_expression as data_exp
import miRNA_nmf_v5 as infer
import miRNA_draw as draw
import miRNA_score as score
import miRNA_module as module
import miRNA_score as score
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
    miRNA = np.divide(miRNA-mean, np.sqrt(var)) + 3*np.ones((1,1))
    mean, var = getStat(miRNA, nInput)
    mRNA_mean, mRNA_var = getStat(mRNA, nOutput)
    mRNA = np.divide(mRNA, mRNA_mean) + 10*np.ones((1,1))
    mRNA_mean, mRNA_var = getStat(mRNA, nOutput)
    
    return miRNA, mRNA, mRNA_mean, mRNA_var
#-----------------------------------------------------------------------------------------------------    


if __name__ == '__main__':

    dataFile = config.Config.all_data_save
    
    miRNA, mRNA = [], []
    if not os.path.isfile(dataFile): 
    # if True:
        if config.Config.key == 'SYNTHETIC':
            nData = config.Config.num_sample
            nInput = config.Config.num_mirna
            nOutput = config.Config.num_mrna
            
            miRNA, mRNA, miRNA_list, mRNA_list, gt, gt_mi, gt_m, putative, gt_sign, ppi, corr, aux = syn.genData(\
                nData,\
                nInput,\
                nOutput,\
                config.Config.num_module,\
                config.Config.num_items_per_module,\
                config.Config.num_putative_per_mirna,\
                config.Config.signal_strength,\
                config.Config.variance_mirna,\
                config.Config.variance_mrna)
        else:
            gene_dic, _ = gene.parse_gene()
            print("loading expressions")
            d = data_exp.ExpData(gene_dic)
            miRNA_list, mRNA_list, miRNA, mRNA = d.getAllExp(mi_th=0.1, gene_th=0.1)
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
            gt_sign = score.convert_corr2sign(corr, gt, 0.001)
            
        if config.Config.save_enable == "Enable":
            with open(dataFile, 'wb') as fdat:
                pickle.dump(
                    (
                        miRNA, mRNA, nInput, nOutput, miRNA_list, mRNA_list,
                        nData, gt, gt_mi, gt_m, putative, corr, ppi, aux, gt_sign
                    ),
                    fdat
                )  
    else:
        f = open(dataFile, "rb")
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        miRNA, mRNA, nInput, nOutput, miRNA_list, mRNA_list, nData, gt, gt_mi, gt_m, putative, corr, ppi, aux, gt_sign = u.load()

    miRNA, mRNA, mRNA_mean, mRNA_var = normalize(miRNA, mRNA, nInput, nOutput)        
    
    # print putative
    # print ppi
    # print gt    

    open('miRNA_list.txt', 'w').write('\n'.join(miRNA_list))
    open('mRNA_list.txt', 'w').write('\n'.join(mRNA_list))
    #print(nInput, nOutput)
    #print("miRNA: ", np.shape(miRNA))
    #print("mRNA: ", np.shape(mRNA))
    #print("corr: ", np.shape(corr))
    #print("putative: ", np.shape(putative))         
    #print("gt: ", np.shape(gt))          
    #print("ppi: ", np.shape(ppi))
    exit(0)

    params = {
        'summary_active': False,
        'learning_rate': 0.01,
        'M': nInput,
        'N': nOutput,
        'K': nInput // 5,
        'num_v': nOutput,
        'num_u': nInput,
        'T': 0.95,
        'I_Phi': putative,
        'I_Omega': ppi,
        'X': miRNA,
        'Y': mRNA,
        'y_var': mRNA_var,
        'y_mean': mRNA_mean,
        'n_sample': nData,
        'n_batch': 256,        
        'n_epoch': 1000,
        'miRNA_names': miRNA_list,
        'mRNA_names': mRNA_list,
        'gt': [gt_mi, gt_m, putative, corr, gt],
        'V_initial': None,
        'U_initial': None
    }    

    S,W,U,V = infer.train(params)
    estimation = S * W * putative
    estimation_sgn = np.copysign(np.ones(estimation.shape), estimation).astype(int)
    estimation_sgn[estimation == 0] = 0

    np.save('OUTPUT_estimation.npy', estimation)
    np.save('OUTPUT_U.npy', U)
    np.save('OUTPUT_V.npy', V)

    from miRNA_module2 import find_modules
    modules = find_modules(miRNA_list, mRNA_list, U, V, 0.5)
    from json import dump
    with open('OUTPUT_modules.json', 'w') as f:
        dump(modules, f)

    #p, r, f, a, tp, fp, tn, fn = score.calcScore2(estimation_sgn, gt_sign, 0, putative)
    #print(('Precision = {}\nRecall = {}\nF1-Score = {}\nAccuracy = {}\n' +
    #        'TP = {}\nFP = {}\nTN = {}\nFN = {}').format(p, r, f, a, tp, fp, tn, fn))
    ##draw.plot_heatmap(params, S*W*putative,U,V, S=S, corr=corr, gt=gt, putative=putative)
        
    
