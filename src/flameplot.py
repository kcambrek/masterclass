""" This function provides...

	A= flameplot(data, <optional>)

 INPUT:
   data:           datamatrix
                   rows    = features
                   colums  = samples
 OPTIONAL

   verbose=        Boolean [0,1]
                   [0]: No (default)
                   [1]: Yes

 OUTPUT
	output

 DESCRIPTION
   Short description what your function does and how it is processed

 EXAMPLE
   %reset -f
   import sys, os, importlib
   
   print(os.getcwd())
   import flameplot
   

   import pandas as pd
   data1 = pd.read_csv('coord2D.csv', sep=';').values
   data2 = pd.read_csv('coord6D.csv', sep=';').values

   out = flameplot(data1, data2, nn=50, steps=2)

 SEE ALSO

print(__doc__)
"""

#--------------------------------------------------------------------------
# Name        : flameplot.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------

#from matplotlib.pyplot import plot
import numpy as np

#%%
def flameplot(data1, data2, nn=250, steps=1, showfig=1, verbose=1):
	#%% DECLARATIONS
    out   = []
    Param = {}
    Param['verbose'] = verbose
    Param['showfig'] = showfig
    Param['steps']   = steps
    Param['nn']      = nn

    #%% Libraries
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    from showprogress import showprogress

    #%% Compute distances
    data1Dist = squareform(pdist(data1,'euclidean'))
    data2Dist = squareform(pdist(data2,'euclidean'))
    
    #%% Take NN based for each of the sample
    data1Order = K_nearestneighbors(data1.T, data1Dist, Param['nn'])
    data2Order = K_nearestneighbors(data2.T, data2Dist, Param['nn'])
    
    #%% Update nn
    Param['nn'] = np.minimum(Param['nn'], len(data1Order[0]))
    Param['nn'] = np.arange(1, Param['nn']+1, Param['steps'])

    #%% Compute overlap
    out = np.zeros((len(Param['nn']),len(Param['nn'])))

    for p in range(0,len(Param['nn'])):
        out[p,:] = overlap_comparison(data1Order, data2Order, Param['nn'], data1.shape[0], Param['nn'][p])
        showprogress(p,len(Param['nn']))
    #end

    #%% Heatmap
    if Param['showfig']==1:
        from imagesc import imagesc
        imagesc(np.flipud(out), cmap='jet', labxrow=Param['nn'], labxcol=Param['nn'], caxis=[0,1])
    #end
    
    #%% END
    return(out)

#%% Take NN based on the number of samples availble
def overlap_comparison(data1Order, data2Order, nn, samples, p):
    out = np.zeros((len(nn),1), dtype='float').ravel()
    for k in range(0,len(nn)):
        tmpoverlap = np.zeros((samples,1), dtype='uint32').ravel()
        
        for i in range(0,samples):
            tmpoverlap[i] = len(np.intersect1d(data1Order[i][0:p], data2Order[i][0:nn[k]]))
        #end
        out[k] = sum(tmpoverlap) / ( len(tmpoverlap) * np.minimum(p,nn[k]) )
    #end
    return(out)

#%% Take NN based on the number of samples availble
def K_nearestneighbors(data1, data1Dist, K):
#    output      = np.zeros((data1.shape[0],data1.shape[0]), dtype='uint32')
#    outputDist  = np.zeros((data1.shape[0],data1.shape[0]), dtype='float')
    outputOrder = []

    # Find closest samples
    for i in range(0, data1Dist.shape[0]):
        I     = np.argsort(data1Dist[i,:])
        Dsort = data1Dist[i,I]
        idx   = np.where(Dsort!=0)[0]
        Dsort = Dsort[idx]
        I     = I[idx]
        I     = I[np.arange(0,np.minimum(K,len(I)))]

        # Store data
#        output[i,I]     = 1
#        outputDist[i,I] = Dsort[np.arange(0,np.minimum(K,len(I)))]
        outputOrder.append(I[np.arange(0,np.minimum(K,len(I)))])
    #end
    return(outputOrder)