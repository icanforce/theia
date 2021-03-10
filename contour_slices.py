from matplotlib import pyplot
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D


def contour_slices(figure, x, y, z, w, cmap=None, norm=None):
    assert x.shape == y.shape == z.shape == w.shape and len(w.shape) == 3

    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(w)
    facecolors = mappable.get_cmap()((norm if norm else lambda a: a)(w))
    ax = figure.gca(projection='3d')

    for i in range(w.shape[2]):
        ax.plot_surface(x[:, :, i], y[:, :, i], z[:, :, i],
                        facecolors=facecolors[:, :, i])

    return mappable


#
# # Example:
#
# # from contour_slices import contour_slices
# import numpy as np
# from matplotlib.cm import jet
#
# x, y, z = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01), np.array([0, 0.1, 0.2, 0.3]))
# w = np.vectorize(lambda x, y, z: 2 / 3 * np.sqrt(np.abs(2 * x**2 + y**3 - 3 * z)))(x, y, z)
# fig = pyplot.figure()
# contour_slices(fig, x, y, z, w, jet)
# pyplot.show()
#
