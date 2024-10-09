""" Density Based clustering.

	A= HDBSCAN(data, <optional>)

 INPUT:
   data:           numpy datamatrix numerical
                   rows    = features
                   colums  = samples
 OPTIONAL

  methodtype=      string: Define your method type
                   'hdbscan' (default)
                   'dbscan'

  min_cluster_size= Integer: Minimum cluster size (only for hdbscan)
                   [2] (default)

  min_samples=     Integer: [0..1] Percentage of expected outliers among number of samples.
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

 OUTPUT
	output

 DESCRIPTION
   http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
   http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
   https://github.com/scikit-learn-contrib/hdbscan
   
   
 EXAMPLE
   %reset -f
   import sys, os, importlib
   
   print(os.getcwd())
   import HDBSCAN
   
   from scatter import scatter
   import numpy as np

   EXAMPLE 1
   from sklearn.datasets.samples_generator import make_blobs
   [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)
   [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)
   scatter(X[:,0],X[:,1], labx=labels_true, labxtype='unique')

   out = HDBSCAN(X)
   scatter(X[:,0],X[:,1], labx=out['labx'], labxtype='unique')
   scatter(X[:,0],X[:,1], size=-np.log10(out['outlier'])*100, labx=out['labx'], labxtype='unique',title='Outliers')
   scatter(X[:,0],X[:,1], size=-np.log10(out['p'])*1000, labx=out['labx'], labxtype='unique',title='Probabilities')

   
   EXAMPLE 2
   from tsneBH import tsneBH
   from UMAPet import UMAPet
   from sklearn.datasets import load_iris
   iris = load_iris()
   X    = tsneBH(iris.data)
   X    = UMAPet(iris.data)
   out = HDBSCAN(X)
   out = HDBSCAN(X, norm=0)
   scatter(X[:,0],X[:,1], size=100, labx=out['labx'])


 SEE ALSO
   import sklearn.cluster as cluster
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
def HDBSCAN(data, min_samples=0.01, metric='euclidean', norm=1, n_jobs=-1, min_cluster_size=2, showprogress=1, showfig=1):
	#%% DECLARATIONS
    out={}
    # Make dictionary to store Parameters
    Param                     = {}
    Param['min_samples']      = min_samples
    Param['min_cluster_size'] = min_cluster_size
    Param['metric']           = metric
    Param['n_jobs']           = n_jobs
    Param['norm']             = norm
    Param['showprogress']     = showprogress
    Param['showfig']     = showfig

    Param['gen_min_span_tree'] = False
    Param['width'] = 10
    Param['height'] = 10

    #%% Libraries
    from sklearn import metrics
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import hdbscan
    import seaborn as sns
    import matplotlib.pyplot as plt

    #%% Set max. outliers
    Param['min_samples'] = np.int(np.floor(Param['min_samples']*data.shape[0]))

    #%% Transform data
    if Param['norm']==1:
        data = StandardScaler().fit_transform(data)
    #end
        
    #%% SET PARAMTERS FOR DBSCAN
    db = hdbscan.HDBSCAN(algorithm='best', metric=Param['metric'], min_samples=np.int(Param['min_samples']), core_dist_n_jobs=Param['n_jobs'], min_cluster_size=np.int(Param['min_cluster_size']), p=None,gen_min_span_tree=Param['gen_min_span_tree'])
    db.fit(data) # Perform the clustering

    out['labx']                = db.labels_        # Labels
    out['p']                   = db.probabilities_ # The strength with which each sample is a member of its assigned cluster. Noise points have probability zero; points in clusters have values assigned proportional to the degree that they persist as part of the cluster.
    out['cluster_persistence'] = db.cluster_persistence_ # A score of how persistent each cluster is. A score of 1.0 represents a perfectly stable cluster that persists over all distance scales, while a score of 0.0 represents a perfectly ephemeral cluster. These scores can be guage the relative coherence of the clusters output by the algorithm.
    out['outlier']             = db.outlier_scores_      # Outlier scores for clustered points; the larger the score the more outlier-like the point. Useful as an outlier detection technique. Based on the GLOSH algorithm by Campello, Moulavi, Zimek and Sander.
    # out2['predict'] = db.prediction_data_     # Cached data used for predicting the cluster labels of new or unseen points. Necessary only if you are using functions from hdbscan.prediction (see approximate_predict(), membership_vector(), and all_points_membership_vectors()).

    #%% Show figures
    if Param['showfig']:
        if Param['min_cluster_size']==True:
            plt.subplots(figsize=(Param['width'],Param['height']))
            db.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
        #end
        plt.subplots(figsize=(Param['width'],Param['height']))
        db.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

        plt.subplots(figsize=(Param['width'],Param['height']))
        db.condensed_tree_.plot()

        plt.subplots(figsize=(Param['width'],Param['height']))
        db.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    #end

    #%% Some info
    if Param['showprogress']:
        n_clusters = len(set(out['labx'])) - (1 if -1 in out['labx'] else 0)
        print('Estimated number of clusters: %d' % n_clusters)

        if n_clusters!=data.shape[0] and n_clusters>1:
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, out['labx']))
        #end
    #end
        
    #%% END
    return(out)
