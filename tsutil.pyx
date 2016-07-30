cimport cython
import numpy as np
cimport numpy as np
import bisect


cdef inline double wmid(long bidp, long bids, long askp, long asks, long tw):
    if (bids+asks)<=0:
        return -1
    if (askp-bidp>tw):
        return (bidp+askp)/2.
    return (bidp*asks+askp*bids)*1.0 / (bids+asks)

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

cdef inline double abz(double a) : return a if a >= 0. else -1 * a
cdef inline long labz(long a) : return a if a >= 0 else -1 * a


include "timestamps.pyx"
include "tsreturns.pyx"
#include "tsvis.pyx"
include "ou.pyx"
include "core.pyx"
include "streaker.pyx"



@cython.cdivision(True)
@cython.boundscheck(False)
def fix_times(np.ndarray[long, ndim=1] df_times):
    return fix_timestamps(df_times)
