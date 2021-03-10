from cluster_comparison import create_pairing, adjusted_rand_score_sets
from miRNA_nmf_v5 import train
from miRNA_module2 import find_modules
from miRNA_score import calcScore2
from miRNA_synthetic_data_v3 import generate_data
from multiprocessing import Pool
from numpy import copysign, mean, ones, set_printoptions, sum, var
import miRNA_tiresias as tiresias
from pvectorize import pvectorize

set_printoptions(threshold=9999, linewidth=9999)

n_sample = 1000
num_miRNA = 50
num_mRNA = 500
num_module = 10

def run(signal, noise, thresh=0.1):
    print('Generating data.... signal={}, noise={}'.format(signal, noise))
    miRNA, mRNA, miRNA_list, mRNA_list, \
    gt, gt_mi, gt_m, putative, gt_sign, ppi, corr, \
    (U, V, true_modules) = generate_data(n_sample, num_miRNA, num_mRNA, num_module, signal, noise, seed=0)

    nStep = 100
    nBatch = 256
    learningRate = 0.01
    rho = sum(1 * (gt > 0)) / float(num_mRNA * num_mRNA)

    #print('nStep=', nStep)
    #print('n_sample=', n_sample)
    #print('nBatch=', nBatch)
    #print('num_mRNA=', num_mRNA)
    #print('num_miRNA=', num_miRNA)
    #print('gt_mi=', gt_mi.shape)
    #print('gt_m=', gt_m.shape)
    #print('putative=', putative.shape)

    tiresias.init(nStep, n_sample, nBatch, num_miRNA, num_mRNA, miRNA, mRNA,
                  [mean(mRNA, axis=0), var(mRNA, axis=0)],
                  miRNA_list, mRNA_list, [gt_mi, gt_m, putative, corr], True)
    loss, ret = tiresias.runTraining(learningRate, rho, preTrain = False)
    p, r, f, a, tp, fp, tn, fn = calcScore2(ret, gt_sign, thresh, putative)            
    print(('Precision = {}\nRecall = {}\nF1-Score = {}\nAccuracy = {}\n' +
           'TP = {}\nFP = {}\nTN = {}\nFN = {}').format(p, r, f, a, tp, fp, tn, fn))

    return f, 0, p, r 


def run_grid():
    from datetime import datetime
    from numpy import arange, meshgrid, vectorize, save
    from pathlib import Path
    from os import chdir

    start = datetime.now()
    s, n = meshgrid(arange(0, 1, 0.05), arange(0, 2, 0.05))
    f_results, ari_results = pvectorize(run)(s, n)

    dirname = 'Tiresias_synthetic_results'
    Path(dirname).mkdir(exist_ok=True)
    chdir(dirname)
    suffix = '-' + start.isoformat() + '.npy'

    save('signal' + suffix, s)
    save('noise' + suffix, n)
    save('f1score' + suffix, f_results)
    save('ari' + suffix, ari_results)

#y1 = []
#y2 = []
#for num_module in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]:
#    num_miRNA = 5 * num_module
#    num_mRNA = 50 * num_module
#    print('')
#    print('num_module = {}'.format(num_module))
#    f, ari = run(0.5, 0.5)
#    y1.append(f)
#    y2.append(ari)
#
#print(y1)
#print(y2)

from numpy import arange
fresults, ariresults, presults, rresults = zip(*(run(0.1, 1.0, thresh=thresh) for thresh in arange(0, 0.25, 0.01)))
print(presults)
print(rresults)
