#---------------------------------------------------------------------
import numpy as np
import pickle
import os
import re
import config
#---------------------------------------------------------------------
# Protein-protein interaction DB downloaded from https://thebiogrid.org/download.php
# e.g., BIOGRID-ALL-3.4.155.tab2.txt
sourceFile = config.Config.ppi_source
dataFile = config.Config.ppi_save
#---------------------------------------------------------------------
def sort_pair(pair):
    pair.sort()
    return tuple(pair)
#---------------------------------------------------------------------
def make_dict():
    dicFile = 'BIOGRID_dic.tmp'
    dict = {}
    if not os.path.isfile(dicFile):    
        with open(sourceFile, 'r') as fd:
            cnt = 0
            for line in fd:
                term = filter(None, re.split('\t', line))
                if len(term) >= 24:
                    # print term[7:11]
                    print(term [18])
                    a = [term[7]]
                    b = [term[8]]
                    a_syn = filter(lambda x: x != '-', re.split('\|', term[9]))
                    b_syn = filter(lambda x: x != '-', re.split('\|', term[10]))
                    # print a_syn, b_syn
                    a = a + a_syn
                    b = b + b_syn
                    # print a, b
                    
                    for aa in a:
                        for bb in b:
                            pair = [aa, bb]
                            pair = sort_pair(pair[:])
                            
                            dict[pair] = 1
                    
                    
                    cnt +=1
                    if (cnt % 100 == 1):
                        print(cnt)
                    
        with open(dicFile, 'wb') as fdat:
            pickle.dump(dict, fdat)
    else:
        print("loading...")
        dict = pickle.load( open( dicFile, "rb" ) )    
        print("done.")
    return dict
#--------------------------------------------------------------------- 
def get_interaction(mRNA_list):
    N = len(mRNA_list)
    interaction = np.zeros((N, N))
    dict = make_dict()
    for i in range(N):
        interaction[i, i] = 1.
        for j in range(i, N):
            pair = [mRNA_list[i], mRNA_list[j]]
            pair = sort_pair(pair[:])
            if pair in dict: #dict.has_key(pair):
                print(pair)
                interaction[i, j] = 1.
                interaction[j, i] = 1.
                
    with open(dataFile, 'wb') as fdat:
        pickle.dump(interaction, fdat)                
    return interaction
            
    

#---------------------------------------------------------------------             
# if __name__ == '__main__':
    # import config
    # config.Config.select_menu()    
    # import miRNA_load_expression_v2 as data_exp    
    
    # global sourceFile, dataFile
    # sourceFile = config.Config.ppi_source
    # dataFile = config.Config.ppi_save    
    
    # d = data_exp.ExpData()
    # miRNA_list, mRNA_list, miRNA, mRNA = d.getAllExp(th=0.01) 
    
    # interaction = get_interaction(mRNA_list)
    
    # print interaction
