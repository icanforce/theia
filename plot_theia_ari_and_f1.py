from matplotlib import pyplot, rc
from numpy import arange, array, delete, load, mean, meshgrid
from os import chdir
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import zoom

s, n = meshgrid(arange(0, 1, 0.05), arange(0, 2, 0.05))
remove_idx = array(arange(-16, 0, 1)) + s.shape[1]
s2 = delete(s, remove_idx, axis=1)
n2 = delete(n, remove_idx, axis=1)
levels = arange(0, 1.1, 0.05)

chdir('Theia_synthetic_results/run2/')
theia_ari = mean(array([
    load('ari-2018-05-22T16:28:39.091974.npy'), 
    load('ari-2018-05-22T16:28:46.575316.npy'),
    load('ari-2018-05-22T16:28:54.529097.npy')
]), axis=0)
theia_f1score = load('f1score-2018-05-22T16:28:39.091974.npy')
chdir('../../')

sigma = 0.5
theia_ari = gaussian_filter(theia_ari, sigma=sigma)
theia_f1score = gaussian_filter(delete(theia_f1score, remove_idx, axis=1), sigma=sigma)

rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
rc('text', usetex=True)

f, axes = pyplot.subplots(1, 2, figsize=(4.5, 1.75))
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)

(ax1, ax2) = axes
cax = ax1.contourf(s, n, theia_ari, levels=levels, cmap='jet')
ax1.set_title(r'ARI')
ax2.contourf(s2, n2, theia_f1score, levels=levels, cmap='jet')
ax2.set_title(r'$\mathrm{F_1}$ score')
ax2.axes.get_yaxis().set_visible(False)

# Add colorbar and axis labels
f.colorbar(cax, ax=axes.ravel().tolist())
ax = f.add_subplot(1, 1, 1, frameon=False)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel(r'Signal strength ($p_{\mathrm{signal}}$)', labelpad=15)
ax.xaxis.set_label_coords(0.4, -0.2)
ax.set_ylabel(r'Relative false positive rate ($p_{\mathrm{fp}}$)', labelpad=15)

pyplot.savefig(r'Theia_ARI_and_F1.eps', bbox_inches='tight', dpi=100)
