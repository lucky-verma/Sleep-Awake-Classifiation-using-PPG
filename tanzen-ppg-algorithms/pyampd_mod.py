# Adapted from pyampd algorithm for Automatic Multi-scale Peak Detection. 
# Adds valley detection as well. Author: Su-Ann Ho

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend

def ampd(x, scale=None, debug=False):
    """Find peaks and valleys in quasi-periodic noisy signals using AMPD algorithm.
    Extended implementation handles peaks near start/end of the signal.
    Optimized implementation by Igor Gotlibovych, 2018
    Parameters
    ----------
    x : ndarray
        1-D array on which to find peaks
    scale : int, optional
        specify maximum scale window size of (2 * scale + 1)
    debug : bool, optional
        if set to True, return the Local Scalogram Matrix, `LSM`,
        weigted number of maxima, 'G',
        and scale at which G is maximized, `l`,
        together with peak locations
    Returns
    -------
    pks: ndarray
        The ordered array of peak indices found in `x`

    valleys: ndarray
        The ordered array of valley indices found in `x`
    """
    x = detrend(x)
    N = len(x)
    L = N // 2
    if scale:
        L = min(scale, L)

    # create LSM matix
    LSM = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        LSM[k - 1, 0:N - k] &= (x[0:N - k] > x[k:N])  # compare to right neighbours
        LSM[k - 1, k:N] &= (x[k:N] > x[0:N - k])  # compare to left neighbours
    
    LSM2 = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        LSM2[k - 1, 0:N - k] &= (x[0:N - k] < x[k:N])  # compare to right neighbours
        LSM2[k - 1, k:N] &= (x[k:N] < x[0:N - k])  # compare to left neighbours

    # Find scale with most maxima
    G = LSM.sum(axis=1)
    G = G * np.arange(
        N // 2, N // 2 - L, -1
    )  # normalize to adjust for new edge regions
    l_scale = np.argmax(G)

    G2 = LSM2.sum(axis=1)
    G2 = G2 * np.arange(
        N // 2, N // 2 - L, -1
    )  # normalize to adjust for new edge regions
    l_scale2 = np.argmax(G2)

    # find peaks that persist on all scales up to l
    pks_logical = np.min(LSM[0:l_scale, :], axis=0)
    pks = np.flatnonzero(pks_logical)
    if debug:
        return pks, LSM, G, l_scale

    #find valleys that persist on all scales up to 1
    valleys_logical = np.min(LSM2[0:l_scale2, :], axis=0)
    valleys = np.flatnonzero(valleys_logical)
    return pks, valleys