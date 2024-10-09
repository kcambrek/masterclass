""" UMAP Uniform Manifold Approximation and Projection (UMAP).

	A= UMAPet(data, <optional>)

 INPUT:
   data:           numpy datamatrix numerical
                   rows    = features
                   colums  = samples
 OPTIONAL

   components=     Integer: Number of components for feature reduction
                   [2]: (default)

   metric=         string: Define your input data as type [metrics.pairwise.calculate_distance] or a distance matrix if thats the case!
                   'euclidean' (default)
                    * euclidean
                    * manhattan
                    * chebyshev
                    * minkowski
                    * canberra
                    * braycurtis
                    * mahalanobis
                    * wminkowski
                    * seuclidean
                    * cosine
                    * correlation
                    * haversine
                    * hamming
                    * jaccard
                    * dice
                    * russelrao
                    * kulsinski
                    * rogerstanimoto
                    * sokalmichener
                    * sokalsneath
                    * yule
        
   n_neighbors=     integer : number of neighboring points used in local approximations of manifold structure
                    Sensible values are in the range 5 to 50, with [10-15] being a reasonable default
                    [5] (default)

   min_dist=       integer:controls how tightly the embedding is allowed compress points together
                    Sensible values are in the range 0.001 to 0.5, with 0.1 being a reasonable default
                    [0.1] (default)
                    [0.5]  embedded points are more evenly distributed
                    [0.001]  optimise more accurately with regard to local structure

   random_state    Integer, Initialization. Note that different initializations might result in different local minima of the cost function.
                   None (default)
                   [0]: Providing a random will give reproducable results

   showprogress=   Boolean [0,1]
                   [0]: No 
                   [1]: Some information about embedding
                   [2]: More information about embedding (default)

 OUTPUT
	output

 DESCRIPTION
   https://github.com/lmcinnes/umap
   
   
 EXAMPLE
   %reset -f
   import sys, os, importlib
   
   print(os.getcwd())
   import UMAPet
   
   import numpy as np

   from sklearn.datasets.samples_generator import make_blobs
   [data, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)

   out = UMAPet(data,showprogress=2)
   scatter(out[:,0],out[:,1], size=100, labx=labels_true)

 SEE ALSO
   pca, tsneBH
   
   pip install umap-learn

"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : UMAP.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------

#%%
def UMAPet(data, components=2, n_neighbors=5, metric='euclidean', min_dist=0.1, random_state=None, showprogress=1):
	#%% DECLARATIONS
    out =[]
    # Make dictionary to store Parameters
    Param = {}
    Param['components']   = components
    Param['n_neighbors']  = n_neighbors
    Param['min_dist']     = min_dist
    Param['metric']       = metric
    Param['random_state'] = random_state
    Param['showprogress'] = showprogress

    #%% Libraries
    import umap
    
    #%% SET PARAMTERS FOR UMAP
    umapP = umap.UMAP(n_components=Param['components'], n_neighbors=Param['n_neighbors'], metric=Param['metric'], min_dist=Param['min_dist'], random_state=Param['random_state'])

    #%% Perform the reduction
#    out1 = tsne.get_params(deep=True)
    out = umapP.fit_transform(data)
    #embed_pca = pca.fit_transform(df)

    #%% END
    return(out)
