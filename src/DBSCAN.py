""" Density Based clustering.

	A= DBSCAN(data, <optional>)

 INPUT:
   data:           numpy datamatrix numerical
                   rows    = features
                   colums  = samples
 OPTIONAL

  eps=             Float: The maximum distance between two samples for them to be considered as in the same neighborhood.
                   [None] (default) Determine automatically by the Siloutte score
                   [0.3] 

  epsres=          Integer: Resoultion to test the different epsilons. The higher the longer it will take
                   [100] (default) 

  minclusters=     Integer: Minimum or more number of clusters >=
                   [2] (default)

  maxclusters=     Integer: Maximum or less number of clusters <=
                   [25] (default)

  min_samples=     Integer: [0.,,1] Percentage of expected outliers among number of samples.
                   [0.05] (default)

  metric=          string: Define your input data as type [metrics.pairwise.calculate_distance] or a distance matrix if thats the case!
                   'euclidean' (default) squared euclidean distance
                   'precomputed' if input is a distance matrix!

  norm=            Boolean [0,1] (You may want to set this =0 using distance matrix as input)
                   [1]: Yes (default) 
                   [0]: No

  n_jobs=          Integer: The number of parallel jobs to run
                   [-1] ALL cpus (default)
                   [1]  Use a single core
                   
  showprogress=   Boolean [0,1]
                   [0]: No 
                   [1]: Some information about embedding
                   [2]: More information about embedding (default)

  showfig=         Boolean [0,1]
                   [0]: No 
                   [1]: Yes (default)

 OUTPUT
	output

 DESCRIPTION
   http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
   http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
   
   
 EXAMPLE
   %reset -f
   import sys, os, importlib
   
   print(os.getcwd())
   import DBSCAN
   
   from scatter import scatter

   EXAMPLE 1
   from sklearn.datasets.samples_generator import make_blobs
   [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)
   [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)

   out = DBSCAN(X)
   scatter(X[:,0],X[:,1], size=100, labx=out['labx'])

   
   EXAMPLE 2
   from sklearn.datasets import load_iris
   iris = load_iris()
   X=iris.data

   from tsneBH import tsneBH
   X   = tsneBH(iris.data)
   out = DBSCAN(X)
   scatter(X[:,0],X[:,1], size=100, labx=out['labx'])
   scatter(X[:,0],X[:,1], size=100, labx=iris.target, labxtype='unique',title='REAL')

   from UMAPet import UMAPet
   X    = UMAPet(iris.data)
   out = DBSCAN(X)
   scatter(X[:,0],X[:,1], size=100, labx=out['labx'])
   scatter(X[:,0],X[:,1], size=100, labx=iris.target, labx_type='unique',title='REAL')


 SEE ALSO
   HDBSCAN, import sklearn.cluster as cluster
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : DBSCAN.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------

#%%
def DBSCAN(data, eps=None, min_samples=0.01, metric='euclidean', norm=1, n_jobs=-1, minclusters=2, maxclusters=25, showfig=1, epsres=100, showprogress=1):
	#%% DECLARATIONS
    out={}
    # Make dictionary to store Parameters
    Param                 = {}
    Param['showprogress'] = showprogress
    Param['showfig']      = showfig
    Param['eps']          = eps
    Param['min_samples']  = min_samples
    Param['metric']       = metric
    Param['n_jobs']       = n_jobs
    Param['norm']         = norm
    Param['minclusters']  = minclusters
    Param['maxclusters']  = maxclusters
    Param['epsres']       = epsres # Resolution of the epsilon to estimate % The higher the more detailed, the more time it costs to compute. Only for DBSCAN

    #%% Libraries
    import sklearn.cluster as cluster
    from sklearn.metrics import silhouette_score
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from silhouette_plot import silhouette_plot
    from showprogress import showprogress
    
    #%% Set max. outliers
    Param['min_samples'] = np.floor(min_samples*data.shape[0])

    #%% Transform data
    if Param['norm']==1:
        data = StandardScaler().fit_transform(data)

    #%% SET PARAMTERS FOR DBSCAN
    if Param['eps']==None:
        if Param['showprogress']: print('Determining optimal clustering by Silhoutte score..')
        # Setup resolution
        eps       = np.arange(0.1,5,1/Param['epsres'])
        silscores = np.zeros(len(eps))*np.nan
        sillclust = np.zeros(len(eps))*np.nan
        silllabx  = []

        # Run over all Epsilons
        for i in range(len(eps)):
            # DBSCAN
            db = cluster.DBSCAN(eps=eps[i], metric=Param['metric'], min_samples=Param['min_samples'], n_jobs=Param['n_jobs']).fit(data)
            # Get labx
            labx=db.labels_

            # Fill array
            sillclust[i]=len(np.unique(labx))
            # Store all labx
            silllabx.append(labx)
            # Compute silhoutte only if more then 1 cluster
            if sillclust[i]>1:
                silscores[i]=silhouette_score(data, db.labels_)
            #end
            showprogress(i,len(eps))
        #end
        
        #%% Convert to array
        silllabx = np.array(silllabx)
        
        #%% Store only if agrees to restriction of input clusters number
        I1 = np.isnan(silscores)==False
        I2 = sillclust>=Param['minclusters']
        I3 = sillclust<=Param['maxclusters']
        I = I1 & I2 & I3

        #%% Get only those of interest
        silscores = silscores[I]
        sillclust = sillclust[I]
        eps       = eps[I]
        silllabx  = silllabx[I,:]
        idx       = np.argmax(silscores)
        
        #%% Plot
        if Param['showfig']==1:
            [fig, ax1] = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(eps, silscores, color='k')
            ax1.set_xlabel('eps')
            ax1.set_ylabel('Silhoutte score')
            ax2.plot(eps, sillclust, color='b')
            ax2.set_ylabel('#Clusters')
            # Plot vertical line To stress the cut-off point
            ax2.axvline(x=eps[idx], ymin=0, ymax=sillclust[idx], linewidth=2, color='r')
        #end
        
        # Store results
        Param['eps'] = eps[idx]
        out['labx']  = silllabx[idx,:]

    else:
        db = cluster.DBSCAN(eps=Param['eps'], metric=Param['metric'], min_samples=Param['min_samples'], n_jobs=Param['n_jobs'])
        db.fit(data)
        # Labels
        out['labx']=db.labels_
    #end


    #%% Some info
    if Param['showprogress']:
        n_clusters = len(set(out['labx'])) - (1 if -1 in out['labx'] else 0)
        print('Estimated number of clusters: %d' % n_clusters)

        if n_clusters!=data.shape[0] and n_clusters>1 and Param['showfig']==1:
            silhouette_plot(data,out['labx'])
        #end
    #end
        
    #%% END
    return(out)
