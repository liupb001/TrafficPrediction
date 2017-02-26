import numpy as np

def computeError(dref, data, sectors):
    RMSEBySector = np.zeros(shape=sectors, dtype=float)
    MAPEBySector = np.zeros(shape=sectors, dtype=float)

    # TODO : size of dref and data

    for i in range(0, sectors):
        a = dref[:, i]
        b = data[:, i]
        rm = np.sqrt((a - b) ** 2)
        ma = np.absolute((a - b) / a) * 100.0
        rm[a <= 0.1] = 0
        ma[a <= 0.1] = 0
        RMSEBySector[i] = np.mean(rm)
        MAPEBySector[i] = np.mean(ma)  # (1.0 / T) * np.sum(np.absolute((a - b) / a)) * 100.0

    meanRMSE = np.mean(RMSEBySector)
    meanMAPE = np.mean(MAPEBySector)
    return RMSEBySector, MAPEBySector

def fixTime(r,m, h):
    # add h x 0 at the beginning, so times are aligned between different horizons
    r= np.hstack([np.zeros(h), r])
    m= np.hstack([np.zeros(h), m])
    return r,m