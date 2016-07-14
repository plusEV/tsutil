import numpy as np
cimport numpy as np
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
def fix_timestamps(np.ndarray[long, ndim=1] df1_times):
    cdef:
        long t_len = df1_times.shape[0]
        long i =0, current_time
        dict time_dict = dict()
        np.ndarray[long, ndim=1] res = np.zeros(t_len, dtype=np.int64)
    for i from 0<= i < t_len:
        current_time = df1_times[i]
        if time_dict.has_key(current_time): #omg its already in here
            res[i] = current_time + time_dict[current_time]
            time_dict[current_time]+=1
        else:
            res[i] = current_time
            time_dict[current_time] = 1
    return res