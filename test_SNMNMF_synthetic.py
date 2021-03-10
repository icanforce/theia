from cluster_comparison import create_pairing, adjusted_rand_score_sets
from miRNA_module2 import find_modules
from miRNA_score import calcScore2
from miRNA_synthetic_data_v3 import generate_data
from multiprocessing import Pool
from numpy import copysign, mean, ones, set_printoptions, var
from SNMNMF_train import Trainer 

set_printoptions(threshold=9999, linewidth=9999)

n_sample = 1000
num_miRNA = 50
num_mRNA = 500
num_module = 10

def run(signal, noise, thresh=0):
    print('Generating data.... signal={}, noise={}'.format(signal, noise))
    miRNA, mRNA, miRNA_list, mRNA_list, \
    gt, gt_mi, gt_m, putative, gt_sign, ppi, corr, \
    (U, V, true_modules) = generate_data(n_sample, num_miRNA, num_mRNA, num_module, signal, noise, seed=0)

    #print('gt_sign=', gt_sign)
    #print('putative=', putative)
    #print('ppi=', ppi)
    #print('corr=', corr)
    #print('true_modules=', true_modules)

    params = {
        'summary_active': False,
        'M': num_miRNA,
        'N': num_mRNA,
        'K': num_module,
        'lambda1': 0.0001,
        'lambda2': 0.01,
        'gamma1': 5,
        'gamma2': 2,
        'I_Phi': putative,
        'I_Omega': ppi,
        'X': miRNA,
        'Y': mRNA,
        'y_var': var(mRNA),
        'n_sample': n_sample,
        'n_iteration': 1000,
        'threshold': thresh,
        'miRNA_names': miRNA_list,
        'mRNA_names': mRNA_list,
        'gt': [gt_mi, gt_m, putative, corr, gt]
    }    

    print('Running SNMNMF...')
    S, U, V = Trainer(params).train()
    #print(S)
    #print(U)
    #print(V)
    modules = find_modules(miRNA_list, mRNA_list, U, V, 0) 
    #print(modules)
    estimation = S * putative
    estimation_sgn = copysign(ones(estimation.shape), estimation).astype(int)
    estimation_sgn[estimation == 0] = 0

    #print('U=', U)
    #print('V=', V)
    #print('S=', S)
    #print('W=', W)
    #print('estimation=', estimation)
    #print('estimation_sgn=', estimation_sgn)
    #print('modules=', modules)

    p, r, f, a, tp, fp, tn, fn = calcScore2(estimation_sgn, gt_sign, 0, putative)
    print(('Precision = {}\nRecall = {}\nF1-Score = {}\nAccuracy = {}\n' +
            'TP = {}\nFP = {}\nTN = {}\nFN = {}').format(p, r, f, a, tp, fp, tn, fn))

    true_modules_combined = [[str(x) for x in m[0]] + ['G' + str(x) for x in m[1]] for m in true_modules]
    modules_combined = [[str(x) for x in m[0]] + ['G' + str(x) for x in m[1]] for m in modules]
    pairing = list(create_pairing(true_modules_combined, modules_combined))
    ari = adjusted_rand_score_sets(true_modules_combined, modules_combined, pairing)
    print('ARI = ', ari)

    return f, ari, p, r


def run_grid():
    from datetime import datetime
    from numpy import arange, meshgrid, vectorize, save
    from pathlib import Path
    from os import chdir

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

    dirname = 'SNMNMF_synthetic_results'
    Path(dirname).mkdir(exist_ok=True)
    chdir(dirname)
    suffix = '-' + start.isoformat() + '.npy'

    save('signal' + suffix, s)
    save('noise' + suffix, n)
    save('f1score' + suffix, f_results)
    save('ari' + suffix, ari_results)


if __name__ == '__main__':
    #run_grid()
    run(0.1, 1)
    pass


#x = list(range(5, 101, 5))
#for num_module in x:
#    num_miRNA = 5 * num_module
#    num_mRNA = 50 * num_module
#    run(0.5, 0.5) 

#from numpy import arange
#fresults, ariresults, presults, rresults = zip(*Pool().starmap(run, ((0.1, 1.0, thresh) for thresh in arange(0, 1, 0.01))))
#print(presults)
#print(rresults)
