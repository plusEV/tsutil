cimport numpy as np
import numpy as np
import pandas as pd

import seaborn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def mech_markets_chart(pnls,volumes,labels,num_sds=4):
    #styling
    fig = plt.figure(figsize=(8,6))
    seaborn.set_style("whitegrid")
    
    pg = seaborn.PairGrid(pnls[0], x_vars=["Time"], y_vars=["PnL", "Exchange Volume"], 
                  aspect=4, palette="coolwarm")
 
    palette = seaborn.color_palette("coolwarm",len(pnls)+1)
    
    c = seaborn.plotting_context("talk", rc={"lines.linewidth": 1})
    mylegend = dict()
    
    #top chart
    for i,p in enumerate(pnls):
        p = p.dropna()
        seaborn.violinplot(p,color=palette[i],widths=.5,ax=pg.axes[0][0],alpha=.45,positions=0,label=labels[i],linewidth=.25)
        with c:
            seaborn.pointplot(p.columns,p.mean(),ax=pg.axes[0][0],x_order=p.columns,color=palette[i],hline=0,label=labels[i]);
        mylegend[labels[i]] = mpatches.Patch(color=palette[i])    
    
    pg.axes[0][0].set_ylabel("PnL(BPs)")
    pg.axes[0][0].set_xlabel("")
    sds = pnls[0].iloc[:,0].std()*num_sds
    pg.axes[0][0].set_ylim((-1*sds,sds))
    
    
    #volume chart
    for i,v in enumerate(volumes):
        v = v.dropna()
        with c:
            seaborn.pointplot(v.columns,v.mean(),ax=pg.axes[1][0],x_order=v.columns,color=palette[i],hline=0,label=labels[i]);
    pg.axes[1][0].set_ylabel("Exchange Volume Diff")
    pg.axes[1][0].set_xlabel("Time Since Trade")
    
    #grids
    pg.axes[0][0].grid(True, which='minor')
    pg.axes[1][0].grid(True, which='minor')
    myticks = map(lambda x: str(x)+'s',pnls[0].columns/1000.)
    #g.set(xticks=(np.arange(15),myticks,rotation=45))
    pg.set(xticklabels=myticks)
    plt.xticks(rotation=45)
    pg.add_legend(mylegend)
    seaborn.despine(left=True)