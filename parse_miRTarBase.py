import numpy as np
import pickle
import os
import re
import config
#---------------------------------------------------------------------
# ground truth files made from http://mirtarbase.mbc.nctu.edu.tw/php/search.php#disease
#---------------------------------------------------------------------
thCnt = config.Config.thCnt
sourceFile = config.Config.gt_source
#---------------------------------------------------------------------
def parse_groundTruth(miRNA_list, mRNA_list):
    pair = []
    #---------------------------------------------------------------------
    with open(sourceFile, 'r') as fd:
        for line in fd:
            term = list(filter(None, re.split(',|\t| \t|\r\n', line)))
            if int(term[6]) >= thCnt:
                pair.append( (term[3],term[4]) )
    #---------------------------------------------------------------------
    ground_truth = np.zeros((len(miRNA_list),len(mRNA_list)))
    gt_mi = []
    gt_m = []
    for i in range(len(miRNA_list)):
        for j in range(len(mRNA_list)):
            p = (miRNA_list[i], mRNA_list[j])
            if p in pair:
                ground_truth[i,j] = 1.
                gt_mi.append(i)
                gt_m.append(j)
                
    return ground_truth, gt_mi, gt_m

