import numpy as np
import pickle
import os
import re
import scipy.stats

import config
from multiprocessing import Pool

miRnaFile_source = config.Config.mirna_exp_source
mRnaFile_source = config.Config.mrna_exp_source
miRnaFile_save = config.Config.mirna_save
mRnaFile_save = config.Config.mrna_save
selected_save = config.Config.selected_exp_save

# miRnaFile_source = 'BRCA_miRNA.csv'
# mRnaFile_source = 'BRCA_gene.csv'
# miRnaFile_save = 'BRCA_miRNA_exp.dat'
# mRnaFile_save = 'BRCA_mRNA_exp.dat'
# selected_save = 'BRCA_selected_exp.txt'

def _helper(exp_miRNA, exp_gene, i, j):
    if j % 10000 == 0:
        print('corr i={}, j={}'.format(i, j))
    r, p = scipy.stats.pearsonr(exp_miRNA[:,i], exp_gene[:,j])
    return r

class ExpData:
    def __init__(self, gene_dic):
        self.exp_miRNA = []
        self.exp_gene = []
        self.miRNA_list = []
        self.gene_list = []    
        self.gene_dic = gene_dic
        
    def readMicroRNA(self):
        if not os.path.isfile(miRnaFile_save): 
            with open(miRnaFile_source, 'r') as fd:
                i = 0
                for line in fd:
                    case = list(filter(None, re.split(',|\r\n', line)))
                    if i == 0:
                        self.miRNA_list = case[1:]
                        i = 1                    
                        continue
                    e = np.asarray(case[1:]).astype(np.float32)
                    e = np.nan_to_num(e)
                    self.exp_miRNA.append(e)
                self.exp_miRNA = np.stack(self.exp_miRNA)
                
            with open(miRnaFile_save, 'wb') as fdat:
                pickle.dump([self.exp_miRNA, self.miRNA_list], fdat)
        else:
            with open(miRnaFile_save, 'rb') as fdat:
                self.exp_miRNA, self.miRNA_list = pickle.load(fdat, encoding='latin1')
        print('miRNA loading done')

    def readGene(self):
        if not os.path.isfile(mRnaFile_save): 
            with open(mRnaFile_source, 'r') as fd:
                i = 0
                for line in fd:
                    case = list(filter(None, re.split(',|\r\n', line)))
                    if i == 0:                    
                        self.gene_list = case[1:]
                        i = 1
                        continue
                    e = np.asarray(case[1:]).astype(np.float32)
                    e = np.nan_to_num(e)
                    self.exp_gene.append(e)
                self.exp_gene = np.stack(self.exp_gene)

            with open(mRnaFile_save, 'wb') as fdat:
                pickle.dump([self.exp_gene, self.gene_list], fdat)
        else:
            with open(mRnaFile_save, 'rb') as fdat:
                self.exp_gene, self.gene_list = pickle.load(fdat, encoding='latin1')
        print('mRNA loading done')
        
    def filtering(self, mi_th, gene_th):
        m_mi = np.mean(self.exp_miRNA, axis=0)
        mi_remove = np.where(m_mi < mi_th)[0]
        m_gene = np.mean(self.exp_gene, axis=0)
        gene_remove = np.where(m_gene < gene_th)[0]
        print("Remove %d, %d: remain %d, %d" % (len(mi_remove), len(gene_remove), len(m_mi)-len(mi_remove), len(m_gene)-len(gene_remove)))
        
        self.exp_miRNA = np.delete(self.exp_miRNA, mi_remove, axis=1)
        self.miRNA_list = np.delete(self.miRNA_list, mi_remove)
        
        self.exp_gene = np.delete(self.exp_gene, gene_remove, axis=1)
        self.gene_list = np.delete(self.gene_list, gene_remove)
        
        #gene_remove = np.where(~np.asarray(list(map(lambda x: x in self.gene_dic, self.gene_list))))[0]
        #self.exp_gene = np.delete(self.exp_gene, gene_remove, axis=1)
        #self.gene_list = np.delete(self.gene_list, gene_remove)
        
        print("Remove %d: remain %d" % (len(gene_remove), len(self.gene_list)))
        
    def getAllExp(self, mi_th=0.01, gene_th=0.01):
        self.readMicroRNA()
        self.readGene()
        self.filtering(mi_th, gene_th)        
        return self.miRNA_list, self.gene_list, self.exp_miRNA, self.exp_gene
        
    def calcCorr(self):
        num_mi = len(self.miRNA_list)
        num_m = len(self.gene_list)
        print('num_mi={}, num_m={}'.format(num_mi, num_m))
        corr = np.zeros((num_mi, num_m))
        
        #for i in range(num_mi):    
        #    print("corr: ", i)
        #    for j in range(num_m):    
        #        r, p = scipy.stats.pearsonr(self.exp_miRNA[:,i], self.exp_gene[:,j])
        #        corr[i, j] = r 
        it = ((self.exp_miRNA, self.exp_gene, i, j) for i in range(num_mi) for j in range(num_m))
        corr = np.fromiter(Pool().starmap(_helper, it), dtype=float).reshape((num_mi, num_m))
        return corr

# if __name__ == '__main__':
    # global miRnaFile_source, mRnaFile_source, miRnaFile_save, selected_save, selected_save
    # config.Config.select_menu()    
    # miRnaFile_source = config.Config.mirna_exp_source
    # mRnaFile_source = config.Config.mrna_exp_source
    # miRnaFile_save = config.Config.mirna_save
    # mRnaFile_save = config.Config.mrna_save
    # selected_save = config.Config.selected_exp_save        
    # exp = ExpData()
    # exp.getAllExp()
        
        
