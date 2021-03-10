from itertools import chain, repeat, tee, starmap
from miRNA_module2 import find_modules
from multiprocessing import Pool 
from numpy import arange, array, copy, cumsum, fill_diagonal, \
    fromiter, insert, matmul, sqrt, sum, where, zeros
from numpy.random import choice, normal, randint, randn, shuffle
from numpy.random import seed as set_random_seed
from scipy.stats import pearsonr


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def flatmap(func, *iterable):
    return chain(*map(func, *iterable))


def randn_skew(n, alpha=0.0):
    sigma = alpha / sqrt(1.0 + alpha * alpha) 
    u0 = randn(n)
    u1 = (sigma * u0 + sqrt(1.0 - sigma * sigma) * randn(n))
    u1[u0 < 0] *= -1
    return u1


def matrix_from_modules(modules, N):
    M = zeros((N, len(modules)))
    for i, m in enumerate(array(m) for m in modules):
        if m.shape[0] > 0:
            M[m, i] = 1
    return M 


def generate_modules(items, modules):
    ns = (1 + randn_skew(items, alpha=5)).astype(int)
    ns[ns < 0] = 0
    ps = fromiter(flatmap(repeat, *zip(*enumerate(ns))), dtype=int)
    shuffle(ps)
    sizes = ((items / modules) + 10 * randn_skew(modules, alpha=5)).astype(int)
    indices = insert(cumsum(sizes), 0, 0)
    ret = list(ps[b:e].tolist() for b, e in pairwise(indices) if b < len(ps))
    return ret + [] * (modules - len(ret))


def generate_modules_u_v(M, N, K):
    modules_m, modules_n = generate_modules(M, K), generate_modules(N, K)
    true_modules = list(zip(modules_m, modules_n))
    m = min(len(modules_m), len(modules_n))
    U, V = matrix_from_modules(modules_m[:m], M), matrix_from_modules(modules_n[:m], N) 
    return true_modules, U, V


def regulate(s, j, n_sample, M, N, signal, ground_truth_sign, x):
    return normal(signal * sum(ground_truth_sign[:, j] * x[s, :]), 1)


def generate_x_y_corr(n_sample, M, N, signal, ground_truth_sign, parallel=False, generate_corr=False):
    x = randn(n_sample, M)

    if parallel: 
        with Pool(processes=4) as p:
            y = array(p.starmap(regulate, ((s, j, n_sample, M, N, signal, ground_truth_sign, x)
                for s in range(n_sample) for j in range(N)))).reshape((n_sample, N))
    else:
        y = array(list(starmap(regulate, ((s, j, n_sample, M, N, signal, ground_truth_sign, x)
                for s in range(n_sample) for j in range(N))))).reshape((n_sample, N))

    corr = None
    if generate_corr: 
        corr = fromiter(
            (pearsonr(x[:, i], y[:, j])[0]
                for j in range(N)
                    for i in range(M)), dtype=float).reshape((M, N))

    return x, y, corr


def generate_data(n_sample, M, N, K, \
    signal=1.0, noise=0.0, up_to_down_ratio=0.5, seed=None):

    if seed is not None:
        set_random_seed(seed) 

    true_modules, U, V = generate_modules_u_v(M, N, K)
    ground_truth = (matmul(U, V.transpose()) > 0) * 1
    density = ground_truth[ground_truth > 0].shape[0] / ground_truth.size
    gt_mi, gt_m = where(ground_truth > 0)
    M, N = ground_truth.shape
    ppi = (matmul(V, V.transpose()) > 0) * 1
    fill_diagonal(ppi, 1)

    prediction = fromiter(
        (choice([0, 1], p=[1 - noise * density, noise * density]) if x < 1 else 1
            for row in ground_truth for x in row),
        dtype=int).reshape(ground_truth.shape)

    ground_truth_sign = fromiter(
        (choice([1, -1], p=[up_to_down_ratio, 1 - up_to_down_ratio]) if x > 0 else 0
            for row in ground_truth for x in row),
        dtype=int).reshape(ground_truth.shape)

    x, y, corr = generate_x_y_corr(n_sample, M, N, signal, ground_truth_sign)

    return \
        x, y, arange(M), arange(N), \
        ground_truth, gt_mi, gt_m, \
        prediction, ground_truth_sign, \
        ppi, corr, (U, V, true_modules)
