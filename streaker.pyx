
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

cpdef inline double kwmid(double bidp, double bids, double askp, double asks,double tick_width):
    if bids < 1 or asks < 1:
        return np.NaN
    if (askp - bidp) > 10 * tick_width:
        return np.NaN    
    if (askp - bidp) > tick_width:
        return (bidp+askp)/2
    else:
        return (bidp*asks+askp*bids)/(bids+asks)

@cython.cdivision(True)
@cython.boundscheck(False)
def nearby_streaker(np.ndarray[double,ndim=2] md, np.ndarray[long,ndim=1] times, long window_size, long cap = 100):
    """
    Times MUST be a unique list of timestamps as LONGS. Prices should be 
    ZERO or NaN if no trade at that tick. Same is true for volumes. Window
    size is in NANOSECONDS.
    
    bp0,bz0,ap0,az0,tp,tz,strike
    """
    cdef:
        long t_len = times.shape[0]
        long md_len = md.shape[0]
        
        long i =0, win_size = window_size, t_diff, j, window_start
        double tots_buys,tots_sells,ref
        
        double this_strike, strike_diff
        dict start_prices = {}
        dict buys = {}
        dict sells = {}
        np.ndarray[double, ndim=2] res = np.zeros((md_len,10), dtype=np.double)
   
    assert(t_len==md_len)
    buy_streaks = [{}]
    sell_streaks = [{}]
    
    for i in range(1,t_len):
        window_start = times[i] - win_size
        j = i
        this_strike = md[i,6]
        
        start_prices = {}
        buys = {}
        sells = {}
        #climb back to find the start of the window
        while times[j]>= window_start and j>=0 and j >= (i-cap):
            start_prices[md[j,6]] = kwmid(md[j,0],md[j,1],md[j,2],md[j,3],.01)
            j-=1
        
        #now we've got starting mid price of all products (assuming they were in the window)
        j+=1

        #now step FORWARD through to calculate the streaks
        while j<=i:
            if md[j,4] > 0 and start_prices.has_key(md[j,6]): #it's a trade
                if md[j,5]>0 and md[j,4]>=start_prices[md[j,6]]: #it's a buy at or above ref
                    if buys.has_key(md[j,6]):
                        buys[md[j,6]] += md[j,5]
                    else:
                        buys[md[j,6]] = md[j,5]
                elif md[j,5]>0 and md[j,4]<=start_prices[md[j,6]]: #it's a sell at or below ref
                    if sells.has_key(md[j,6]):
                        sells[md[j,6]] += md[j,5]
                    else:
                        sells[md[j,6]] = md[j,5]
                    
            j+=1

        for k in buys.keys():
            strike_diff = this_strike - k
            if (strike_diff == 5):
                res[i,4] = buys[k]
            if (strike_diff == 2.5):
                 res[i,3] = buys[k]
            if (strike_diff == 0):
                 res[i,2] = buys[k]     
            if (strike_diff == -2.5):
                 res[i,1] = buys[k]  
            if (strike_diff == -5):
                 res[i,0] = buys[k]  
        
        for k in sells.keys():
            strike_diff = this_strike - k
            if (strike_diff == 5):
                res[i,9] = sells[k]
            if (strike_diff == 2.5):
                 res[i,8] = sells[k]
            if (strike_diff == 0):
                 res[i,7] = sells[k]     
            if (strike_diff == -2.5):
                 res[i,6] = sells[k]  
            if (strike_diff == -5):
                 res[i,5] = sells[k]

    return res

@cython.cdivision(True)
@cython.boundscheck(False)
def nearby_wmid_changes(np.ndarray[double,ndim=2] md, np.ndarray[long,ndim=1] times, long window_size, long cap = 100):
    """
    Times MUST be a unique list of timestamps as LONGS. Prices should be 
    ZERO or NaN if no trade at that tick. Same is true for volumes. Window
    size is in NANOSECONDS.
    
    bp0,bz0,ap0,az0,tp,tz,strike
    """
    cdef:
        long t_len = times.shape[0]
        long md_len = md.shape[0]
        
        long i =0, win_size = window_size, t_diff, j, window_start
        
        double this_strike, strike_diff
        dict start_prices = {}
        dict finish_prices = {}
        np.ndarray[double, ndim=2] res = np.zeros((md_len,5), dtype=np.double)
   
    assert(t_len==md_len)
    
    for i in range(1,t_len):
        window_start = times[i] - win_size
        j = i
        this_strike = md[i,6]
        
        start_prices = {}
        buys = {}
        sells = {}
        #climb back to find the start of the window
        while times[j]>= window_start and j>=0 and j >= (i-cap):
            start_prices[md[j,6]] = kwmid(md[j,0],md[j,1],md[j,2],md[j,3],.01)
            j-=1
        
        #now we've got starting mid price of all products (assuming they were in the window)
        j+=1

        #now step FORWARD through to calculate the wmid changes
        while j<=i:
            finish_prices[md[j,6]] = kwmid(md[j,0],md[j,1],md[j,2],md[j,3],.01)    
            j+=1
            
        for k in finish_prices.keys():
            strike_diff = this_strike - k
            if not start_prices.has_key(k):
                continue
            if (strike_diff == 5):
                res[i,4] = finish_prices[k] - start_prices[k]
            if (strike_diff == 2.5):
                 res[i,3] = finish_prices[k] - start_prices[k]
            if (strike_diff == 0):
                 res[i,2] = finish_prices[k] - start_prices[k]     
            if (strike_diff == -2.5):
                 res[i,1] = finish_prices[k] - start_prices[k]  
            if (strike_diff == -5):
                 res[i,0] = finish_prices[k] - start_prices[k]
    return res