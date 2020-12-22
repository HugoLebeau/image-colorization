import meshio
import pkg_resources
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

mesh = meshio.read(pkg_resources.resource_filename(__name__, 'cielab_d65_2_visible.vtk'))
poly = Delaunay(mesh.points[:, [1, 2]])

is_in_visible_gamut = lambda point: poly.find_simplex(point) >= 0

def in_visible_gamut(dq=10, bottom=-200, top=200):
    '''
    Compute a quantization of the a*b* space and keep points that are in the
    visible gamut.

    Parameters
    ----------
    dq : int, optional
        Quantization step (consider a dq x dq square around each point). The
        default is 10.
    bottom : int, optional
        Minimum possible value in the quantization. The default is -200.
    top : int, optional
        Maximum possible value in the quantization. The default is 200.

    Returns
    -------
    ndarray, shape (Q, 2)
        A quantization of the a*b* space with Q points in the visible gamut.

    '''
    l = np.arange(bottom+dq//2, top+1, dq)
    quantized_ab = np.array(np.meshgrid(l, l))
    quantized_ab = np.transpose(quantized_ab, (1, 2, 0))
    return quantized_ab[is_in_visible_gamut(quantized_ab)]

if __name__ == '__main__':
    visible_ab = in_visible_gamut(dq=10)
    pd.DataFrame(visible_ab).to_csv('in_visible_gamut.csv', header=False, index=False)
    
    import matplotlib.pyplot as plt
    plt.scatter(visible_ab[:, 1], visible_ab[:, 0], marker='.')
    plt.axis('equal')
    plt.xlabel("b")
    plt.ylabel("a")
    plt.title("Quantization of the a*b* space")
    plt.show()
