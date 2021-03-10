from matplotlib import pyplot, rc
from numpy import arange, array, delete, load, mean, meshgrid
from os import chdir
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import zoom

s, n = meshgrid(arange(0, 1, 0.05), arange(0, 2, 0.05))
remove_idx = array(arange(-16, 0, 1)) + s.shape[1]
s2 = delete(s, remove_idx, axis=1)
n2 = delete(n, remove_idx, axis=1)
levels = arange(-1, 1.1, 0.1)

chdir('Theia_synthetic_results/run2/')
theia_ari = mean(array([
    load('ari-2018-05-22T16:28:39.091974.npy'), 
    load('ari-2018-05-22T16:28:46.575316.npy'),
    load('ari-2018-05-22T16:28:54.529097.npy')
]), axis=0)
theia_f1score = load('f1score-2018-05-22T16:28:39.091974.npy')
chdir('../')
chdir('../SNMNMF_synthetic_results2/')
snmnmf_rel_ari = load('ari-2018-05-31T23:14:40.184074.npy') - theia_ari
snmnmf_rel_f1score = load('f1score-2018-05-31T23:14:40.184074.npy') - theia_f1score
chdir('../PIMIM_synthetic_results/')
pimim_rel_ari = load('ari-2018-06-01T17:39:14.880930.npy') - theia_ari
pimim_rel_f1score = load('f1score-2018-06-01T17:39:14.880930.npy') - theia_f1score
chdir('../Tiresias_synthetic_results/')
tiresias_rel_f1score = zoom(load('f1score-2018-06-22T15:34:47.370829.npy'), zoom=2, order=0) - theia_ari
chdir('../')

# Compute relative results and apply Gaussian filter
sigma = 0.5
theia_ari = gaussian_filter(theia_ari, sigma=sigma)
theia_f1score = gaussian_filter(delete(theia_f1score, remove_idx, axis=1), sigma=sigma)
snmnmf_rel_ari = gaussian_filter(snmnmf_rel_ari, sigma=sigma)
snmnmf_rel_f1score = gaussian_filter(delete(snmnmf_rel_f1score, remove_idx, axis=1), sigma=sigma)
pimim_rel_ari = gaussian_filter(pimim_rel_ari, sigma=sigma)
pimim_rel_f1score = gaussian_filter(delete(pimim_rel_f1score, remove_idx, axis=1), sigma=sigma)
tiresias_rel_f1score = gaussian_filter(tiresias_rel_f1score, sigma=sigma)
tiresias_rel_f1score_zoomed = delete(tiresias_rel_f1score, remove_idx, axis=1)

rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
rc('text', usetex=True)

f, axes = pyplot.subplots(3, 2, figsize=(4.75, 5.75))
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.2)

((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes
cax = ax1.contourf(s, n, snmnmf_rel_ari, levels=levels, cmap='jet_r')
ax1.set_title(r'SNMNMF relative ARI')
ax1.axes.get_xaxis().set_visible(False)
ax2.contourf(s2, n2, snmnmf_rel_f1score, levels=levels, cmap='jet_r')
ax2.set_title(r'SNMNMF relative $\mathrm{F_1}$ score')
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax3.contourf(s, n, pimim_rel_ari, levels=levels, cmap='jet_r')
ax3.set_title(r'PIMiM relative ARI')
ax3.axes.get_xaxis().set_visible(False)
ax4.contourf(s2, n2, pimim_rel_f1score, levels=levels, cmap='jet_r')
ax4.set_title(r'PIMiM relative $\mathrm{F_1}$ score')
ax4.axes.get_xaxis().set_visible(False)
ax4.axes.get_yaxis().set_visible(False)
ax5.contourf(s, n, tiresias_rel_f1score, levels=levels, cmap='jet_r')
ax5.set_title(r'Tiresias relative $\mathrm{F_1}$ score')
ax6.contourf(s2, n2, tiresias_rel_f1score_zoomed, levels=levels, cmap='jet_r')
ax6.set_title(r'Tiresias relative $\mathrm{F_1}$ score')
ax6.axes.get_yaxis().set_visible(False)

# Add colorbar and axis labels
f.colorbar(cax, ax=axes.ravel().tolist())
ax = f.add_subplot(1, 1, 1, frameon=False)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel(r'Signal strength ($p_{\mathrm{signal}}$)', labelpad=20)
ax.xaxis.set_label_coords(0.4, -0.075)
ax.set_ylabel(r'Relative false positive rate ($p_{\mathrm{fp}}$)', labelpad=20)

pyplot.savefig(r'ARI_and_F1_comparison.eps', bbox_inches='tight', dpi=100)
