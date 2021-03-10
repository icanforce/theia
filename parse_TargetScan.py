import numpy as np
import pickle
import os
import re
import pickle
import config
#---------------------------------------------------------------------
# Download the following two files from http://www.targetscan.org/cgi-bin/targetscan/data_download.vert71.cgi
sourceFile = config.Config.putative_source
dataFile = config.Config.putative_save
# sourceFile = ["Conserved_Site_Context_Scores.txt", "Nonconserved_Site_Context_Scores.txt"]
# dataFile = "BRCA_pred.dat"
#---------------------------------------------------------------------
def load(fname, dict, dict_group):
    cnt1 = 0
    with open(fname, 'r') as fd:
        for line in fd:
            cnt1 += 1
            if cnt1>1:
                if cnt1 % 100 == 0:
                    print(fname, cnt1)
                term = list(filter(None, re.split(',|\t| \t|\r\n', line)))
                # print term
                # print term[1], term[4], term[8]
                mirna = term[4]
                mrna = term[1]
                contextpp = 0
                if mirna not in dict:
                    dict[mirna] = {}
                if term[8] != 'NULL':
                    contextpp = float(term[8])
                dict[mirna][mrna] = contextpp
                
                expr = r'(hsa-[a-zA-Z]{3}-[0-9]+[a-z]?).*'
                m = re.search(expr, mirna, flags=0)
                if m:
                    mirna_rep = m.group(1)
                    if mirna[-2:] == '3p':
                        mirna_rep += r'*3p'
                    elif mirna[-2:] == '5p':
                        mirna_rep += r'*5p'
                    if mirna_rep not in dict_group:
                        dict_group[mirna_rep] = []
                    if not (mirna in dict_group[mirna_rep]):
                        dict_group[mirna_rep].append(mirna)
#--------------------------------------------------------------------- 
def parse_prediction(miRNA, mRNA, useContextpp = False ):
    dict = {}
    dict_group = {}

    if not os.path.isfile(dataFile):    
        for i in range(len(sourceFile)):
            load(sourceFile[i], dict, dict_group)

        with open(dataFile, 'wb') as fdat:
            pickle.dump([dict, dict_group], fdat)
    else:
        print("loading...")
        dict, dict_group = pickle.load( open( dataFile, "rb" ) )    


    prediction_pair = np.zeros( (len(miRNA), len(mRNA)) )

    for i in range(len(miRNA)):
        mRNA_list = []
        mi_name = miRNA[i]
        m = re.search(r'(hsa-[a-zA-Z]{3}-[0-9]+[a-z]?).*', mi_name, flags=0)
        if m:
            mirna_rep = m.group(1)
            if mi_name[-2:] == '3p':
                mirna_rep += r'*3p'
            elif mi_name[-2:] == '5p':
                mirna_rep += r'*5p'        
                
            if mirna_rep in dict_group: #dict_group.has_key(mirna_rep):
                mirna_group = dict_group[mirna_rep]
            elif m.group(1) in dict_group: #dict_group.has_key(m.group(1)):
                mirna_group = dict_group[m.group(1)]
            else:
                print(m.group(1), 'not in the database')
                
            print(mi_name, mirna_group)
            for mi in mirna_group:
                mRNA_list.append( dict[mi] )
            if len(mRNA_list) == 0:
                print("there is no " + mi_name + " in the list")
                continue                
            
            for l in range(len(mRNA_list)):
                for j in range(len(mRNA)):
                    if mRNA[j] in mRNA_list[l]: #mRNA_list[l].has_key(mRNA[j]):
                        print(mirna_group[l], mRNA[j], dict[mirna_group[l]][mRNA[j]])
                        prediction_pair[i, j] = 1
                        if (useContextpp == True):
                            prediction_pair[i, j] = dict[mirna_group[l]][mRNA[j]]
    
    return prediction_pair
