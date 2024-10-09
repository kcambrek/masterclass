""" This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method

	A = silhouette(data, metric='euclidean', linkage='ward')

 INPUT:
   data:           datamatrix
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

 verbose=        Boolean [0,1]: Progressbar
                 [0]: No (default)
                 [1]: Yes

 OUTPUT
	output

 DESCRIPTION
  This function return the cluster labels for the optimal cutt-off based on the choosen clustering method
  
 EXAMPLE
   import silhouette
   from silhouette import silhouette

   from sklearn.datasets.samples_generator import make_blobs
   [data, labels_true] = make_blobs(n_samples=750, centers=5, n_features=10)
   out= silhouette(data)


 SEE ALSO
   silhouette_plot
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : silhouette.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
#--------------------------------------------------------------------------

#%% Libraries
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from showprogress import showprogress
from silhouette_plot import silhouette_plot
import matplotlib.pyplot as plt

#%%
def silhouette(data, metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, width=15, height=8, showfig=1, Z=[], verbose=0):
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


    #%% Cluster hierarchical using on metric/linkage
    if len(Z)==0:
        Z=linkage(data, method=Param['linkage'], metric=Param['metric'])
    

    #%% Make all possible cluster-cut-offs
    if Param['verbose']: print('Determining optimal clustering by Silhoutte score..')

    # Setup storing parameters
    clustcutt = np.arange(Param['minclusters'],Param['maxclusters'])
    silscores = np.zeros((len(clustcutt)))*np.nan
    sillclust = np.zeros((len(clustcutt)))*np.nan
    clustlabx = []

    #%% Run over all cluster cutoffs
    for i in range(len(clustcutt)):
        # Cut the dendrogram for i clusters
        labx = fcluster(Z, clustcutt[i], criterion='maxclust')
        # Store labx for cluster-cut
        clustlabx.append(labx)
        # Store number of unique clusters
        sillclust[i]=len(np.unique(labx))
        # Compute silhoutte (can only be done if more then 1 cluster)
        if sillclust[i]>1:
            silscores[i]=silhouette_score(data, labx)
        
        if Param['verbose'] ==1:  showprogress(i,len(clustcutt))

    

    #%% Convert to array
    clustlabx = np.array(clustlabx)
    
    #%% Store only if agrees to restriction of input clusters number
    I1 = np.isnan(silscores)==False
    I2 = sillclust>=Param['minclusters']
    I3 = sillclust<=Param['maxclusters']
    I  = I1 & I2 & I3

    # Get only clusters of interest
    silscores = silscores[I]
    sillclust = sillclust[I]
    clustlabx = clustlabx[I,:]
    clustcutt = clustcutt[I]
    idx       = np.argmax(silscores)
    
    #%% Plot
    if Param['showfig']==1:
        # Make figure
        [fig, ax1] = plt.subplots(figsize=(Param['width'],Param['height']))
        # Plot
        ax1.plot(sillclust, silscores, color='k')
        # Plot optimal cut
        ax1.axvline(x=clustcutt[idx], ymin=0, ymax=sillclust[idx], linewidth=2, color='r',linestyle="--")
        # Set fontsizes
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rc('xtick', labelsize=10)     # fontsize of the axes title
        plt.rc('ytick', labelsize=10)     # fontsize of the axes title
        plt.rc('font', size=10)
        # Set labels
        ax1.set_xticks(clustcutt)
        ax1.set_xlabel('#Clusters')
        ax1.set_ylabel('Score')
        ax1.set_title("Silhoutte score versus number of clusters")
        ax1.grid(color='grey', linestyle='--', linewidth=0.2)
    

    #%% Plot silhoutte samples plot
    if Param['showfig']==1:
        silhouette_plot(data,clustlabx[idx,:])
    
    
    #%% Store results
    out['score'] = pd.DataFrame(np.array([sillclust,silscores]).T, columns=['clusters','score'])
    out['score'].clusters = out['score'].clusters.astype(int)
    out['labx']  = clustlabx[idx,:]-1
    
    #%% END
    return(out)
