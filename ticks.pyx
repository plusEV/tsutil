from libc.math cimport isnan

EPS = .0001

def tickDir(double last_bid, double last_ask, double cur_bid, double cur_ask):
    if cur_ask > last_ask or cur_bid > last_bid: #soft or hard uptick
        return 1
    elif cur_bid < last_bid or cur_ask < last_ask: #soft downtick
        return -1
    return 0

def tickPrice(double last_bid, double last_ask, double cur_bid, double cur_ask, double ts):
    
    if cur_ask > last_ask: #soft uptick
        return last_ask - ts
    elif cur_bid < last_bid: #soft downtick
        return last_bid + ts
    return 0

def tick_dirs_multi(np.ndarray[object, ndim=1] syms, np.ndarray[double, ndim=2] md, np.ndarray[double, ndim=1] widths):
    cdef:
        long N = md.shape[0]
        long X=0, side
        long i

        dict lasts = {}
        object sym

        double last_bid = 0
        double last_ask = 0
        double prc, last_prc, last_side, width
        cdef np.ndarray[np.double_t, ndim=1] ticks = np.zeros(N,dtype=np.double)
    
    assert(len(syms) == len(md))

    for i in range(N):

        sym = syms[i]
        width = widths[i]
        if not lasts.has_key(sym):
            lasts[sym] = [md[i,0],md[i,1],md[i,0],0]
            continue

        last_bid = lasts[sym][0]
        last_ask = lasts[sym][1]
        last_prc = lasts[sym][2]
        last_side = lasts[sym][3]

        if last_bid != 0 and last_ask != 0:

            prc = tickPrice(last_bid,last_ask, md[i,0],md[i,1],ts)
            side = tickDir(last_bid,last_ask, md[i,0],md[i,1])
            
            if side == 1:
                if not (last_side == 1 and prc <= last_prc):
                    ticks[i] = side
            elif side == -1:
                if not (last_side == -1 and prc >= last_prc):
                    ticks[i] = side
                

        if ticks[i] != 0:
            lasts[sym] = [md[i,0],md[i,1],prc,side]
        else:
            lasts[sym] = [md[i,0],md[i,1],last_prc,last_side]
    return ticks

def tick_dirs(np.ndarray[double, ndim=2] md, double ts = .01):
    cdef:
        long N = md.shape[0]
        long X=0, side
        long i
        double last_bid = 0
        double last_ask = 0
        double prc, last_prc, last_side
        cdef np.ndarray[np.double_t, ndim=1] ticks = np.zeros(N,dtype=np.double)
    
    for i in range(N):
        
        if last_bid != 0 and last_ask != 0:
            prc = tickPrice(last_bid,last_ask, md[i,0],md[i,1],ts)
            side = tickDir(last_bid,last_ask, md[i,0],md[i,1])
            
            if side == 1:
                if not (last_side == 1 and prc <= last_prc):
                    ticks[i] = side
            elif side == -1:
                if not (last_side == -1 and prc >= last_prc):
                    ticks[i] = side
                
        last_bid = md[i,0]
        last_ask = md[i,1]
        if ticks[i] != 0:
            last_prc = prc
            last_side = side

    return ticks

def ticks_to_next_ticks(np.ndarray[double,ndim=1] ticks):
    cdef:
        long N = ticks.shape[0], i = N-1
        double tick, last_tick
        np.ndarray[np.double_t, ndim=1] res = np.zeros(N,dtype=np.double)
    while (i>=0):
        tick = ticks[i]
        res[i] = last_tick
        if tick!=0:
            last_tick = tick
       
        i = i-1
    return res

def ticks_to_prev_ticks(np.ndarray[double,ndim=1] ticks):
    cdef:
        long N = ticks.shape[0], i = 0
        double tick, last_tick
        np.ndarray[np.double_t, ndim=1] res = np.zeros(N,dtype=np.double)
    while (i<N):
        tick = ticks[i]
        res[i] = last_tick
        if tick!=0:
            last_tick = tick
        i = i+1
    return res 



EPS = .0001

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