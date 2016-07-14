cimport numpy as np
cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)
def prev_prices(np.ndarray[object, ndim=1] syms,np.ndarray[long, ndim=1] bids,np.ndarray[long, ndim=1] asks):  
    cdef:
        long slen = syms.shape[0]
        dict last_info = dict()
        np.ndarray[long, ndim=1] bidprices = np.zeros(slen, dtype=np.int64)
        np.ndarray[long, ndim=1] askprices = np.zeros(slen, dtype=np.int64)
    for i from 0<= i < slen:
        if last_info.has_key(syms[i]):
            bidprices[i] = last_info[syms[i]][0]
            askprices[i] = last_info[syms[i]][1]
        last_info[syms[i]] = (bids[i],asks[i])
    return bidprices,askprices

@cython.cdivision(True)
@cython.boundscheck(False)
def trade_prices(np.ndarray[object, ndim=1] syms,np.ndarray[long, ndim=1] bids,np.ndarray[long, ndim=1] asks):  
    cdef:
        long slen = syms.shape[0]
        dict last_info = dict()
        np.ndarray[long, ndim=1] bidprices = np.zeros(slen, dtype=np.int64)
        np.ndarray[long, ndim=1] askprices = np.zeros(slen, dtype=np.int64)
    for i from 0<= i < slen:
        if last_info.has_key(syms[i]):
            bidprices[i] = last_info[syms[i]][0]
            askprices[i] = last_info[syms[i]][1]
            if bids[i]<bidprices[i]: # we downticked
                askprices[i] = bidprices[i]
            elif asks[i]>askprices[i]: #we upticked
                bidprices[i] = askprices[i]
        last_info[syms[i]] = (bids[i],asks[i])
    return bidprices,askprices