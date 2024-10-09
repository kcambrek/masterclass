""" This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method

	A = dbindex(data, metric='euclidean', linkage='ward')

 INPUT:
   data:         datamatrix
                 rows    = features
                 colums  = samples
 OPTIONAL

 metric=         String: Distance measure for the clustering 
                 'euclidean' (default)

 linkage=        String: Linkage type for the clustering 
                 'ward' (default)

 minclusters=    Integer: Minimum or more number of clusters >=
                 [2] (default)

 maxclusters=    Integer: Maximum or less number of clusters <=
                 [25] (default)


 showfig=        Boolean [0,1]: Progressbar
                 [0]: No (default)
                 [1]: Yes (silhoutte plot)
                   
 height=         Integer:  Height of figure
                 [5]: (default)

 width=          Integer:  Width of figure
                 [5]: (default)


 Z=              Object from linkage function. This will speed-up computation if you readily have Z
                 [] (default)
                 Z=linkage(data, method='ward', metric='euclidean')
 
 verbose=        Boolean [0,1]: Progressbar
                 [0]: No (default)
                 [1]: Yes

 OUTPUT
	output

 DESCRIPTION
  This function return the cluster labels for the optimal cutt-off based on the choosen clustering method
  
 EXAMPLE
   %reset -f
   import sys, os, importlib
   
   print(os.getcwd())
   import dbindex
   

   from sklearn.datasets.samples_generator import make_blobs
   [data, labels_true] = make_blobs(n_samples=750, centers=6, n_features=10)
   out= dbindex(data)



 SEE ALSO
   silhouette, silhouette_plot, elbowclust
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : dbindex.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------

#from matplotlib.pyplot import plot
from scipy.spatial.distance import euclidean
import numpy as np

#%%
def dbindex(data, metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, width=15, height=8, showfig=1, Z=[], verbose=1):
	#%% DECLARATIONS
    out ={}
    
    # Make dictionary to store Parameters
    Param = {}
    Param['verbose']     = verbose
    Param['metric']      = metric
    Param['linkage']     = linkage
    Param['minclusters'] = minclusters
    Param['maxclusters'] = maxclusters
    Param['showfig']     = showfig
    Param['height']      = height
    Param['width']       = width

    #%% Libraries
    from scipy.cluster.hierarchy import linkage, fcluster
    import pandas as pd
    from showprogress import showprogress
    import matplotlib.pyplot as plt
    
    #%% Cluster hierarchical using on metric/linkage
    if len(Z)==0:
        Z=linkage(data, method=Param['linkage'], metric=Param['metric'])
    #end
    
    #%% Make all possible cluster-cut-offs
    if Param['verbose']: print('Determining optimal clustering by Silhoutte score..')

    # Setup storing parameters
    clustcutt = np.arange(Param['minclusters'],Param['maxclusters'])
    scores = np.zeros((len(clustcutt)))*np.nan
    dbclust = np.zeros((len(clustcutt)))*np.nan
    clustlabx = []

    #%% Run over all cluster cutoffs
    for i in range(len(clustcutt)):
        # Cut the dendrogram for i clusters
        labx = fcluster(Z, clustcutt[i], criterion='maxclust')
        # Store labx for cluster-cut
        clustlabx.append(labx)
        # Store number of unique clusters
        dbclust[i]=len(np.unique(labx))
        # Compute silhoutte (can only be done if more then 1 cluster)
        if dbclust[i]>1:
            scores[i]=dbindex_score(data, labx)
        #end
        if Param['verbose'] ==1:  showprogress(i,len(clustcutt))

    #end

    #%% Convert to array
    clustlabx = np.array(clustlabx)
    
    #%% Store only if agrees to restriction of input clusters number
    I1 = np.isnan(scores)==False
    I2 = dbclust>=Param['minclusters']
    I3 = dbclust<=Param['maxclusters']
    I  = I1 & I2 & I3

    # Get only clusters of interest
    scores = scores[I]
    dbclust = dbclust[I]
    clustlabx = clustlabx[I,:]
    clustcutt = clustcutt[I]
    idx       = np.argmin(scores)
    
    #%% Plot
    if Param['showfig']==1:
        # Make figure
        [fig, ax1] = plt.subplots(figsize=(Param['width'],Param['height']))
        # Plot
        ax1.plot(dbclust, scores, color='k')
        # Plot optimal cut
        ax1.axvline(x=clustcutt[idx], ymin=0, ymax=dbclust[idx], linewidth=2, color='r',linestyle="--")
        # Set fontsizes
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rc('xtick', labelsize=10)     # fontsize of the axes title
        plt.rc('ytick', labelsize=10)     # fontsize of the axes title
        plt.rc('font', size=10)
        # Set labels
        ax1.set_xticks(clustcutt)
        ax1.set_xlabel('#Clusters')
        ax1.set_ylabel('Score')
        ax1.set_title("Davies Bouldin index versus number of clusters")
        ax1.grid(color='grey', linestyle='--', linewidth=0.2)
    #end
    
    #%% Store results
    out['score'] = pd.DataFrame(np.array([dbclust,scores]).T, columns=['clusters','score'])
    out['score'].clusters = out['score'].clusters.astype(int)
    out['labx']  = clustlabx[idx,:]-1
    
    #%% END
    return(out)

#%% Compute DB-score
def dbindex_score(X, labels):
#    n_cluster = len(np.bincount(labels))
    n_cluster = np.unique(labels)
#    cluster_k = [X[labels == k] for k in range(n_cluster)]
    cluster_k=[]
    for k in range(0, len(n_cluster)):
        cluster_k.append(X[labels==n_cluster[k]])
    #end
#    cluster_k = [X[labels == k] for k in range(n_cluster)]
    
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]

    db = []
    for i in range(0,len(n_cluster)):
        for j in range(0,len(n_cluster)):
            if n_cluster[j] != n_cluster[i]:
                db.append( (variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]) )
            #end
        #end
    #end
    outscore = np.max(db) / len(n_cluster)
    return(outscore)