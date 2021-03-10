from matplotlib import pyplot, rc
from numpy import arange 

w = 0.25

rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
rc('text', usetex=True)

f, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(4.5, 1.75))
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0.1)

y_pos = arange(4) 
ax1.bar(y_pos - w, [112, 88, 48, 37], align='center', color=['tab:blue'] * 4, label='Theia', width=w)
ax1.bar(y_pos, [60, 56, 52, 2], align='center', color=['tab:green'] * 4, label='PIMiM', width=w)
ax1.bar(y_pos + w, [24, 20, 2, 8], align='center', color=['tab:orange'] * 4, label='SNMNMF', width=w)
ax1.set_xticks(y_pos)
ax1.set_xticklabels(['A', 'B', 'C', 'D'])
#ax1.set_title('Module comparison')
ax1.set_ylabel('Number of modules', labelpad=5)

y_pos = arange(3) 
ax2.bar(y_pos - w, [491, 302, 219], align='center', color=['tab:blue'] * 3, label='Theia',  width=w)
ax2.bar(y_pos, [503, 86, 4], align='center', color=['tab:green'] * 3, label='PIMiM', width=w)
ax2.bar(y_pos + w, [20, 18, 141], align='center', color=['tab:orange'] * 3, label='SNMNMF', width=w)
ax2.set_xticks(y_pos)
ax2.set_xticklabels(['E', 'F', 'G'])
#ax2.set_title('MiRNA/gene comparison')
ax2.set_ylabel('Number of miRNAs or genes', labelpad=0)

ax = f.add_subplot(1, 1, 1, frameon=False)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax1.legend(loc=(0.2, -0.3), ncol=3)

pyplot.savefig('BRCA_comparison.eps', bbox_inches='tight', dpi=100)
