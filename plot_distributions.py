from itertools import groupby
from json import load
from matplotlib import pyplot, rc, ticker

def splititer(s, sep):
    start = 0
    seplen = len(sep)
    while True:
        i = s.find(sep, start)
        if i == -1:
            yield s[start:]
            break
        yield s[start:i]
        start = i + seplen

clusters = load(open('../miRBase/hsa-22-clusters-50000b.json'))
miRNA_lens = sorted(len(set(cluster)) for cluster in clusters)

with open('../goatools/human_2018-6-6-modified.assocs') as f:
    from collections import defaultdict
    go2gene = defaultdict(list)
    gene2go = defaultdict(list)
    for line in f:
        gene, go_terms_str = line.split('\t')
        for go_term in splititer(go_terms_str, ';'):
            go2gene[go_term].append(gene)
            gene2go[gene].append(go_term)
    mRNA_lens = sorted(len(c) for c in go2gene.values() if len(c) <= 150)
    mRNA_lens = [x for x in mRNA_lens if x > 1]

#theia = load(open('../co-modules/BRCA-co-modules7-trimmed.json'))
#snmnmf = load(open('../co-modules/SNMNMF-BRCA-modules.json'))
#pimim = load(open('../co-modules/PIMIM-BRCA-modules.json'))
#miRNA_lens, mRNA_lens = zip(*[(len(m[0]), len(m[1])) for m in theia])

rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
rc('text', usetex=True)

f, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(4.5, 1.75))
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0.1)

tick_formatter = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))

# (Theia) 3 outliers are removed from miRNA_lens: 18, 20, 26
ax1.hist(miRNA_lens, range=(1, 16), bins=15, histtype='step', color='r')
ax1.set_xlabel('Number of MiRNAs')
ax1.set_ylabel('Clusters (thousands)')
ax1.yaxis.set_major_formatter(tick_formatter)

ax2.hist(mRNA_lens, range=(1, 151), bins=15, histtype='step', color='b')
ax2.set_xlabel('Number of Genes')
ax2.set_ylabel('GO terms (thousands)')
ax2.yaxis.set_major_formatter(tick_formatter)

#ax = f.add_subplot(1, 1, 1, frameon=False)
#ax.tick_params(which='both', bottom=False, left=False)
#ax.set_xticklabels([])
#ax.set_yticklabels([])

pyplot.savefig("Distribution.eps", bbox_inches='tight', dpi=100)
