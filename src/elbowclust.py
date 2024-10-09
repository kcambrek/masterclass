""" This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method

	A = elbowclust(data, metric='euclidean', linkage='ward')

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
   import elbowclust
   

   from sklearn.datasets.samples_generator import make_blobs
   [data, labels_true] = make_blobs(n_samples=750, centers=6, n_features=10)
   out= elbowclust(data)



 SEE ALSO
   silhouette, silhouette_plot, dbindex, 
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : elbowclust.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------

#from matplotlib.pyplot import plot
import numpy as np

#%%
def elbowclust(data, metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, width=15, height=8, showfig=1, Z=[], verbose=1):
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
    import matplotlib.pyplot as plt
    
    #%% Cluster hierarchical using on metric/linkage
    if len(Z)==0:
        Z=linkage(data, method=Param['linkage'], metric=Param['metric'])
    #end

    #%% Make all possible cluster-cut-offs
    if Param['verbose']: print('Determining optimal clustering by derivatives..')

    #%% Run over all cluster cutoffs
    last     = Z[-10:, 2]
    last_rev = last[::-1]
    idxs     = np.arange(1, len(last) + 1)
    
    acceleration     = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]

    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("clusters: %d" %k)
    
    #%% Now use the optimal cluster cut-off for the selection of clusters
    clustlabx = fcluster(Z, k, criterion='maxclust')

    #%% Convert to array
    clustlabx = np.array(clustlabx)
    
    #%% Plot
    if Param['showfig']==1:
        # Make figure
        [fig, ax1] = plt.subplots(figsize=(Param['width'],Param['height']))
        # Plot
        plt.plot(idxs, last_rev)
        plt.plot(idxs[:-2] + 1, acceleration_rev)
        
        # Plot optimal cut
        ax1.axvline(x=k, ymin=0, linewidth=2, color='r',linestyle="--")
        # Set fontsizes
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rc('xtick', labelsize=10)     # fontsize of the axes title
        plt.rc('ytick', labelsize=10)     # fontsize of the axes title
        plt.rc('font', size=10)
        # Set labels
        ax1.set_xticks(np.arange(0,len(idxs)))
        ax1.set_xlabel('#Clusters')
        ax1.set_ylabel('Score')
        ax1.set_title("Derivatives versus number of clusters")
        ax1.grid(color='grey', linestyle='--', linewidth=0.2)
    #end
    
    #%% Store results
    out['labx'] = clustlabx
    
    #%% END
    return(out)
