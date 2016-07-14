cimport cython
import numpy as np
cimport numpy as np

def bidask_asof(some_df,some_sides):
    """
    For each timestamp in Pandas series sides, return the bid/ask based on side.
    
    Paramters
     @ some_df -- Full dataframe of all symbols, prices
     @ sides -- pandas Series of sides, buys MUST be 1
    
    Returns Pandas Series of prices indexed at sides (NOW AS DOUBLES CAREFUL)
     
    """
    cdef:
        np.ndarray[long, ndim=1] ts = some_sides.index.astype(long)
        np.ndarray[long, ndim=1] sides = (some_sides == 1).astype(long).values
        np.ndarray[double, ndim=1] res = np.zeros(ts.size) * np.NaN
        long t_len = ts.size
    
    for i in range(t_len):
        this_stamp = some_df.index[(bisect.bisect_right(some_df.index,ts[i])-1)]
        if sides[i]:
            res[i] = some_df.ix[this_stamp,'ask1']
        else:
            res[i] = some_df.ix[this_stamp,'bid1']
    return pd.Series(res,index=ts)

def trades_asof(some_df,some_sides):
    """
    For each timestamp in Pandas series sides, return the bid/ask based on side.
    
    Paramters
     @ some_df -- Full dataframe of all symbols, prices
     @ sides -- pandas Series of sides, buys MUST be 1
    
    Returns Pandas Series of prices indexed at sides (NOW AS DOUBLES CAREFUL)
     
    """

    nprices = some_df.tradeprice.replace(0,np.NaN)
    return pd.Series(nprices.asof(some_sides.index).values,index=some_sides.index)    

def run_strat(some_df,theos_frame):
    """
    Given a Pandas DataFrame of theo/side indexed with LONG timestamps, attempt to execute
    vs some_df. (Which should only be MD of the thing you have theos for)
    
    Paramters
     @ some_df -- Full dataframe of all symbols, prices
     @ theo_series -- theos Frame (columns MUST be "theo" and "side" -- buys must 1)
    
    
    Returns Pandas DataFrame of Price/Quantity available to execute
     
    """
    
    cdef:
        np.ndarray[long, ndim=1] ts = theos_frame.index.astype(long)
        np.ndarray[double, ndim=1] theos = theos_frame.theo.values.astype(np.double)
        np.ndarray[long, ndim=1] sides = (theos_frame.side == 1).values.astype(long)
        np.ndarray[double, ndim=2] res = np.zeros((ts.size,2)) * np.NaN
        
        long t_len = ts.size
        long this_stamp, this_pos, stop_stamp, window_stop, i, j, k
    
    for i in range(t_len):
        this_stamp = some_df.index[(bisect.bisect_right(some_df.index,ts[i])-1)]
        if sides[i] and theos[i]>=some_df.ix[this_stamp,'ask1']:#buy
            res[i,0] = some_df.ix[this_stamp,'ask1']
            res[i,1] = some_df.ix[this_stamp,'asksize1']
        elif not sides[i] and theos[i]<=some_df.ix[this_stamp,'bid1']:#sell
            res[i,0] = some_df.ix[this_stamp,'bid1']
            res[i,1] = some_df.ix[this_stamp,'bidsize1']
    return pd.DataFrame(res,index=ts,columns=['price','qty'])