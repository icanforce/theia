from matplotlib import pyplot, rc
from numpy import arange 

w = 0.25

rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
rc('text', usetex=True)

f, ax = pyplot.subplots(1, 1, figsize=(2, 1.75))
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)

theia = [0.5701754385964912, 0.5925925925925926, 0.8]
tiresias = [0.47368421052631576, 0.46296296296296297, 0.6444444444444445]
pimim = [0.07894736842105263, 0.08333333333333333, 0.13333333333333333]

y_pos = arange(3) 
ax.bar(y_pos - w, theia, align='center', color=['tab:blue'] * 4, label='Theia', width=w)
ax.bar(y_pos, tiresias, align='center', color=['tab:red'] * 4, label='Tiresias', width=w)
ax.bar(y_pos + w, pimim, align='center', color=['tab:green'] * 4, label='PIMiM', width=w)
ax.set_xticks(y_pos)
ax.set_xticklabels(['0.001', '0.01', '0.1'])
ax.set_xlabel('PCC', labelpad=10)
ax.set_ylabel('Detection rate', labelpad=10)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

pyplot.savefig('Detection_rate_bar.eps', bbox_inches='tight', dpi=100)
