from cluster_comparison import create_pairing, adjusted_rand_score_sets
from miRNA_nmf_v5 import train
from miRNA_module2 import find_modules
from miRNA_score import calcScore2
from miRNA_synthetic_data_v3 import generate_data
from multiprocessing import Pool
from numpy import copysign, mean, ones, set_printoptions, var
from pvectorize import pvectorize
from datetime import datetime
from numpy import arange, meshgrid, vectorize, save
from pathlib import Path
from os import chdir


set_printoptions(threshold=9999, linewidth=9999)

n_sample = 1000
num_miRNA = 50
num_mRNA = 500
num_module = 10

def run(signal, noise, lambda1=0.5, lambda2=0.5, lambda3=0.25, thresh=0):
    print('Generating data.... signal={}, noise={}, lambda1={}, lambda2={}, lambda3={}'.format(signal, noise, lambda1, lambda2, lambda3))
    miRNA, mRNA, miRNA_list, mRNA_list, \
    gt, gt_mi, gt_m, putative, gt_sign, ppi, corr, \
    (U, V, true_modules) = generate_data(n_sample, num_miRNA, num_mRNA, num_module, signal, noise, seed=0)

    #from numpy import count_nonzero
    #print(count_nonzero(gt) / 25000)
    #print(count_nonzero(putative) / 25000)
    #exit(0)

    #print('gt_sign=', gt_sign)
    #print('putative=', putative)
    #print('ppi=', ppi)
    #print('corr=', corr)
    #print('true_modules=', true_modules)

    params = {
        'summary_active': False,
        'learning_rate': 0.01,
        'M': num_miRNA,
        'N': num_mRNA,
        'K': num_module,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'lambda3': lambda3,
        'num_v': num_mRNA,
        'num_u': num_miRNA,
        'T': 0.95,
        'I_Phi': putative,
        'I_Omega': ppi,
        'X': miRNA,
        'Y': mRNA,
        'y_var': var(mRNA, axis=0),
        'y_mean': mean(mRNA, axis=0),
        'n_sample': n_sample,
        'n_batch': 1000, 
        'n_epoch': 1000,
        'miRNA_names': miRNA_list,
        'mRNA_names': mRNA_list,
        'gt': [gt_mi, gt_m, putative, corr, gt],
        'U_initial': None,
        'V_initial': None 
    }

    print('Running Theia...')
    S, W, U, V = train(params)
    modules = find_modules(miRNA_list, mRNA_list, U, V, 0.5) 
    estimation = S * W * putative
    estimation_sgn = estimation
    #estimation_sgn = copysign(ones(estimation.shape), estimation).astype(int)
    #estimation_sgn[estimation == 0] = 0

    #print('U=', U)
    #print('V=', V)
    #print('S=', S)
    #print('W=', W)
    #print('estimation=', estimation)
    #print('estimation_sgn=', estimation_sgn)
    #print('modules=', modules)

    p, r, f, a, tp, fp, tn, fn = calcScore2(estimation_sgn, gt_sign, thresh, putative)
    print(('Precision = {}\nRecall = {}\nF1-Score = {}\nAccuracy = {}\n' +
            'TP = {}\nFP = {}\nTN = {}\nFN = {}').format(p, r, f, a, tp, fp, tn, fn))

    true_modules_combined = [[str(x) for x in m[0]] + ['G' + str(x) for x in m[1]] for m in true_modules]
    modules_combined = [[str(x) for x in m[0]] + ['G' + str(x) for x in m[1]] for m in modules]
    pairing = list(create_pairing(true_modules_combined, modules_combined))
    ari = adjusted_rand_score_sets(true_modules_combined, modules_combined, pairing)
    print('ARI = ', ari)

    return f, ari, p, r


def run_grid():
    start = datetime.now()

    #run1
    #s, n = meshgrid([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], arange(0, 2, 0.1))
    #run2
    s, n = meshgrid(arange(0, 1, 0.05), arange(0, 2, 0.05))
    f_results, ari_results = vectorize(run)(s, n)

    #print(s)
    #print(n)
    #print(f_results)
    #print(ari_results)

    dirname = 'Theia_synthetic_results2'
    Path(dirname).mkdir(exist_ok=True)
    chdir(dirname)
    suffix = '-' + start.isoformat() + '.npy'

    save('signal' + suffix, s)
    save('noise' + suffix, n)
    save('f1score' + suffix, f_results)
    save('ari' + suffix, ari_results)


def theia_hyperparameter_experiment():
    a, b, l1, l2, l3 = meshgrid(0.1, 1.0, arange(0, 1, 0.05), arange(0, 1, 0.05), arange(0, 1, 0.05))
    dirname = 'Theia_hyperparameter_results'
    Path(dirname).mkdir(exist_ok=False)
    chdir(dirname)
    save('l1.npy', l1)
    save('l2.npy', l2)
    save('l3.npy', l3)
    fresults, ariresults, presults, rresults = pvectorize(run)(a, b, l1, l2, l3)
    save('fresults.npy', fresults)
    save('ariresults.npy', ariresults)
    save('presults.npy', presults)
    save('rresults.npy', rresults)




#from numpy import array
#fresults, ariresults, presults, rresults = vectorize(run)(arange(0, 0.2, 0.01), 0.25)
#print(presults)
#print(rresults)

run(0.1, 1)
#theia_hyperparameter_experiment()

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
#print(y1)
#print(y2)

#fresults, ariresults, presults, rresults = zip(*(run(0.1, 1.0, thresh=thresh) for thresh in arange(0, 0.12, 0.01)))
#print(presults)
#print(rresults)
