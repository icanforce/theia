import config
config.Config.select_menu()
import parse_gene as gene
import miRNA_load_expression as data_exp
from numpy import arange, load, mean
from os import chdir

gene_dic, _ = gene.parse_gene()
d = data_exp.ExpData(gene_dic)
miRNA_list, mRNA_list, miRNA, mRNA = d.getAllExp(mi_th=0.1, gene_th=0.1)

chdir('Theia_BRCA_results')
U = load('OUTPUT_U.npy')
V = load('OUTPUT_V.npy')
assert U.shape[1] == V.shape[1]
K = U.shape[1]

def f(name, M, k, th):
    return [name[i] for i in range(M.shape[0]) if M[i][k] > th]

modules = [(f(miRNA_list, U, k, 0.5), f(mRNA_list, V, k, 0.25)) for k in range(K)]
modules = [m for m in modules if len(m[0]) > 1 and len(m[1]) > 1]
miRNA_modules, mRNA_modules = zip(*modules)
miRNA_lens = sorted(len(x) for x in miRNA_modules)
mRNA_lens = sorted(len(x) for x in mRNA_modules)
print('Number of modules: {}'.format(len(modules)))
print('Mean number of miRNA: {}'.format(mean(miRNA_lens)))
print('Mean number of mRNA: {}'.format(mean(mRNA_lens)))

from json import dump
dump(modules, open('modules.json', 'w'))
