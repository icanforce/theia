g = from_numpy_matrix(ppi)
louvain_mrna_modules = [[e for e, _ in group] for _, group in groupby(sorted(best_partition(g).items(), key=snd), key=snd)]
louvain_modules = [([i for x in m for i in range(putative.shape[0]) if estimation_sgn[i, x] > 0], m) for m in louvain_mrna_modules]
louvain_modules = [(a, b) for a, b in louvain_modules if len(a) > 1 or len(b) > 1]
