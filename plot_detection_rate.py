from matplotlib import pyplot, rc

x = [0.001 * x for x in range(1, 11)]

theia_pcc0001, theia_pcc01 = [0.5714035087719298, 0.5175438596491229, 0.4649122807017544, 0.4649122807017544, 0.42105263157894735, 0.40350877192982454, 0.37719298245614036, 0.32456140350877194, 0.30701754385964913, 0.2899298245614035], [0.8, 0.7111111111111111, 0.6444444444444445, 0.6444444444444445, 0.5777777777777777, 0.5555555555555556, 0.5333333333333333, 0.4666666666666667, 0.4666666666666667, 0.4444444444444444]

tiresias_pcc0001, tiresias_pcc01 = [0.47368421052631576, 0.45614035087719296, 0.43859649122807015, 0.4298245614035088, 0.42105263157894735, 0.42105263157894735, 0.40350877192982454, 0.40350877192982454, 0.40350877192982454, 0.40350877192982454], [0.6444444444444445, 0.6444444444444445, 0.6444444444444445, 0.6444444444444445, 0.6444444444444445, 0.6444444444444445, 0.6222222222222222, 0.6222222222222222, 0.6222222222222222, 0.6222222222222222]

rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
rc('text', usetex=True)

f, (ax1, ax2) = pyplot.subplots(1, 2, sharey=True, figsize=(4.5, 1.75))
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)

ax1.plot(x, theia_pcc01, label='Theia')
ax1.plot(x, tiresias_pcc01, label='Tiresias', color='r')
ax1.set_title(r'Filtering PCC $\geq$ 0.1')
ax1.set_xlabel(r'Threshold ($T_{\mathbf{W}}$)', labelpad=10)

ax2.plot(x, theia_pcc0001, label='Theia')
ax2.plot(x, tiresias_pcc0001, label='Tiresias', color='r')
ax2.set_title(r'Filtering PCC $\geq$ 0.001')
ax2.set_xlabel(r'Threshold ($T_{\mathbf{W}}$)', labelpad=10)
ax2.tick_params(which='both', left=False)

ax = f.add_subplot(1, 1, 1, frameon=False)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_ylabel(r'Detection rate', labelpad=15)
ax1.legend(loc=(0.6, -0.5), ncol=2)

pyplot.savefig('Detection_rate.eps', bbox_inches='tight', dpi=100)
