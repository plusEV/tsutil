
@cython.cdivision(True)
@cython.boundscheck(False)
def streaker(np.ndarray[long,ndim=1] times, np.ndarray[double,ndim=1] prices, 
    np.ndarray[long,ndim=1] volumes,long window_size):
    """
    Times MUST be a unique list of timestamps as LONGS. Prices should be 
    ZERO or NaN if no trade at that tick. Same is true for volumes. Window
    size is in NANOSECONDS.
    """
    cdef:
        long t_len = times.shape[0]
        long s_len = prices.shape[0]
        
        long i =0, win_size = window_size, t_diff, j, window_start
        double tots_buys,tots_sells,ref
        
        np.ndarray[long, ndim=2] res = np.zeros([t_len,2], dtype=long)
   
    assert(t_len==s_len)

    for i in range(1,t_len):
        window_start = times[i] - win_size
        j = i
        tots_buys = 0
        tots_sells = 0
        
        ref = prices[i]
        
        #climb back to find the start of the window
        while times[j]>= window_start and j>=0:
            j-=1
        j+=1
        #now step FORWARD through to calculate the streaks
        while j<=i:
            if prices[j] > 0: #it's a trade

                if volumes[j]>0 and prices[j]>=ref: #it's a buy at or above ref
                    tots_buys+=volumes[j]
                elif volumes[j]<0 and prices[j]<=ref: #it's a sell at or below ref
                    tots_sells+=abz(volumes[j])
                    
            j+=1
        res[i][0] = tots_buys
        res[i][1] = tots_sells
    return res

@cython.cdivision(True)
@cython.boundscheck(False)
def streaker_with_refs(np.ndarray[long,ndim=1] times, np.ndarray[double,ndim=1] prices, 
    np.ndarray[long,ndim=1] volumes,np.ndarray[double,ndim=1] refs, long window_size, long cap = 100):
    """
    Times MUST be a unique list of timestamps as LONGS. Prices should be 
    ZERO or NaN if no trade at that tick. Same is true for volumes. Window
    size is in NANOSECONDS.
    """
    cdef:
        long t_len = times.shape[0]
        long s_len = prices.shape[0]
        
        long i =0, win_size = window_size, t_diff, j, window_start
        double tots_buys,tots_sells,ref
        
        np.ndarray[long, ndim=2] res = np.zeros([t_len,2], dtype=long)
   
    assert(t_len==s_len)

    for i in range(1,t_len):
        window_start = times[i] - win_size
        j = i
        tots_buys = 0
        tots_sells = 0
        
        ref = refs[i]
        
        #climb back to find the start of the window
        while times[j]>= window_start and j>=0 and j >= (i-cap):
            j-=1
        j+=1
        #now step FORWARD through to calculate the streaks
        while j<=i:
            if prices[j] > 0: #it's a trade

                if volumes[j]>0 and prices[j]>=ref: #it's a buy at or above ref
                    tots_buys+=volumes[j]
                elif volumes[j]<0 and prices[j]<=ref: #it's a sell at or below ref
                    tots_sells+=abz(volumes[j])
                    
            j+=1
        res[i][0] = tots_buys
        res[i][1] = tots_sells
    return res