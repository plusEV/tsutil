cimport numpy as np
import numpy as np
import pandas as pd


def ts_returns(some_df,timestamps,tw,theos=pd.Series(),mode='bp', 
                buckets=[-180000,-30000,-10000,-5000,-1000,0,1000,5000,10000,30000,180000], colnames = ['bp0','bz0','ap0','az0']):
    """
    Calculates basis point returns from array of long timestamps indexing into ref frame.
    
    Paramters
     @ some_df -- Full dataframe of all symbols, prices
     @ timestamps -- timestamps to partition around
    
    Default buckets: '-10s','-1s','-100ms','-10ms','-1ms','-100us','-10us','0',
        '+10us','+100us','+1ms','+10ms','+100ms','+1s','+10s'
    
    Buckets are in MILLISECONDS
    
    Returns changes in BASIS POINTS
     
    """
    
    cdef:
        np.ndarray[long, ndim=1] ts = timestamps.astype(long)
        np.ndarray[double, ndim=1] bucks = np.array(buckets).astype(np.double)
        np.ndarray[double, ndim=2] vals = np.zeros((ts.size,len(bucks))) * np.NaN
        long t_len = ts.size, b_len = bucks.size,f_len = some_df.shape[0]
        long this_stamp, this_pos, stop_stamp, window_stop, i, j, k
        long A_MILLI = 1000000 # A MILLI A MILLI
        double b
    
    
    for i in range(t_len):
        this_stamp = some_df.index[(bisect.bisect_right(some_df.index,ts[i])-1)]
        this_pos = some_df.index.get_loc(this_stamp) #position in dataframe of this timestamp
        
        if this_pos >= f_len:
            continue
        
        for j in range(b_len):
            k = this_pos
            b = bucks[j] #this bucket
            if b==0:
                vals[i,j] = wmid(some_df.ix[this_stamp,colnames[0]],some_df.ix[this_stamp,colnames[1]],\
                           some_df.ix[this_stamp,colnames[2]],some_df.ix[this_stamp,colnames[3]],tw) 
                continue
            window_stop = (long)(ts[i] + b * A_MILLI)
            if b<0:
                while some_df.index[k]>=window_stop and k>=0:
                    k-=1
                if k<0:
                    k=0
            else:
                while k < f_len and some_df.index[k]<window_stop:
                    k+=1
                if k>=f_len:
                    k=f_len-1
            stop_stamp = some_df.index[k]
            if (stop_stamp - this_stamp) > (buckets[-1]*2*A_MILLI):
                continue #don't include cross session nonsense
            vals[i,j] = wmid(some_df.ix[stop_stamp,colnames[0]],some_df.ix[stop_stamp,colnames[1]],\
                           some_df.ix[stop_stamp,colnames[2]],some_df.ix[stop_stamp,colnames[3]],tw) 

    
    res = pd.DataFrame(vals,index=pd.to_datetime(timestamps),columns=buckets)
    if len(theos)==res.shape[0]:
        print "Using Theos..."
        res[0.0] = theos.values
    if mode=='bp':
        return ((res.sub(res[0.0],axis=0)) / res[0.0] * 1e4)
    return (res.sub(res[0.0],axis=0))/ tw




def vol_buckets(some_df,timestamps,buckets=[-10000,-1000,-100,-10,-1,-.1,-.01,0,.01,.1,1,10,100,1000,10000], colnames=['lastsize']):
    """
    Calculates contract volume sums indexed off array timestamps into ref frame.
    
    Paramters
     @ some_df -- Full dataframe of all symbols, prices
     @ timestamps -- timestamps to partition around
    
    Default buckets: '-10s','-1s','-100ms','-10ms','-1ms','-100us','-10us','0',
        '+10us','+100us','+1ms','+10ms','+100ms','+1s','+10s'
    
    Buckets are in MILLISECONDS
    
    Returns frame of volumes across buckets
     
    """
    
    cdef:
        np.ndarray[long, ndim=1] ts = timestamps.astype(long)
        np.ndarray[double, ndim=1] bucks = np.array(buckets).astype(np.double)
        np.ndarray[double, ndim=2] vals = np.zeros((ts.size,len(bucks)))
        long t_len = ts.size, b_len = bucks.size,f_len = some_df.shape[0]
        long this_stamp, this_pos, stop_stamp, window_stop, i, j, k, temp
        long A_MILLI = 1000000 # A MILLI A MILLI
        double b
    
    
    for i in range(t_len):
        this_stamp = some_df.index[(bisect.bisect_right(some_df.index,ts[i])-1)]
        this_pos = some_df.index.get_loc(this_stamp) #position in dataframe of this timestamp

        
        for j in range(b_len):
            k = this_pos
            temp = this_pos
            b = bucks[j] #this bucket
            if b==0:
                continue
            window_stop = (long)(ts[i] + b * A_MILLI)
            if b<0:
                while some_df.index[k]>=window_stop and k>=0:
                    k-=1
                if k<0:
                    k=0
            else:
                while some_df.index[k]<window_stop and k<f_len:
                    k+=1
                if k>=f_len:
                    k=f_len-1
            
            #swap before walk forward and summing volume        
            if k>this_pos:
                temp = k
                k = this_pos + 1
            elif temp==0:
                continue
            else:
                temp-=1
            
            
            #now step FORWARD through to calculate the sum volumes
            while k<=temp:
                stop_stamp = some_df.index[k]
                vals[i,j] += labz(some_df.ix[stop_stamp,colnames[0]])
                k+=1
                
    res = pd.DataFrame(vals,index=pd.to_datetime(timestamps),columns=buckets)
    return res
