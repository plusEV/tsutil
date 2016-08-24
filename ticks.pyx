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