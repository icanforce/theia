import numpy as np
import pickle
import os
import re
import csv

import config
gene_source = config.Config.gene_source
gene_key = config.Config.gene_key
# gene_source = "./data/all_gene_disease_associations.tsv"
# gene_key = "breast"

def parse_gene(th=0.1):
    gene = {}
    with open(gene_source, encoding='ISO-8859-1') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            m = re.search('[Bb]reast', row[3], flags=0)  
            if m:
                # print row
                g = row[1]
                if g not in gene:
                    if float(row[4]) > th:
                        gene[g] = 1
    gene_list = gene.keys()
    return gene, len(gene_list)
         

            
    
