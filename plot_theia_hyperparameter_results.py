from contour_slices import contour_slices
from os import chdir
import numpy as np
from matplotlib import pyplot, rc
from matplotlib.cm import jet
from matplotlib.colors import Normalize

rc('text', usetex=True)

chdir('Theia_hyperparameter_results')
x, y, z, w = tuple(np.load(name + '.npy')[0][0] for name in ('l1', 'l2', 'l3', 'ariresults'))

#print('\n'.join('\t'.join(str(round(c, 2)) for c in t)
#                for t in sorted(((x[i][j][k], y[i][j][k], z[i][j][k], w[i][j][k])
#                                 for i in range(w.shape[0])
#                                 for j in range(w.shape[1])
#                                 for k in range(w.shape[2])),
#                                key=lambda x: x[-1], reverse=True)))

i = np.unravel_index(w.argmax(), w.shape)
print('Max: x={}, y={}, z={}'.format(x[i], y[i], z[i]))

for i in sorted(set(range(0, 20)) - {0, 5, 10, 15}, reverse=True):
    x, y, z, w = tuple(np.delete(a, i, axis=2) for a in (x, y, z, w))

rc('text', usetex=True)
fig = pyplot.figure(figsize=(4.5, 2.75))
m = contour_slices(fig, x, y, z, w, jet, norm=Normalize(vmin=0.8, vmax=1))
fig.colorbar(m, pad=0.15)
ax = fig.axes[0]
ax.set_xlabel('$\lambda_1$')
ax.set_ylabel('$\lambda_2$')
ax.set_zlabel('$\lambda_3$')
ax.view_init(20, -45)
pyplot.savefig('hyperparameter_results.eps', bbox_inches='tight', dpi=100)
