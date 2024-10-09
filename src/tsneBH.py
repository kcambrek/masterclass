""" tSNE barnes hut version.

	A= tsneBH(data, <optional>)

 INPUT:
   data:           numpy datamatrix numerical
                   rows    = features
                   colums  = samples
 OPTIONAL

   components=     Integer: Number of components for feature reduction
                   [2]: (default)

   perplexity=     Float
                   30 (default)

   init=           String/numpy array
                   'random' (default)
                   'pca':    Start with initialization derived from PCA. Usually more globally stable than random initialization. However, cannot be used with precomputed distances!
                    numpy array with n_samples, n_components

   method=         string: The gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be used when nearest-neighbor errors need to be better than 3%. However, the exact method cannot scale to millions of examples.
                   'barnes_hut' (default)
                   'exact'      (slow)

   metric=         string: Define your input data as type [metrics.pairwise.calculate_distance] or a distance matrix if thats the case!
                   'euclidean' (default) squared euclidean distance
                   'precomputed' if input is a distance matrix!

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
   tSNE barnes hut version
   http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
   http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
   http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2016/tutorials/aux8_tsne.html
   
   
 EXAMPLE
   %reset -f
   import numpy as np

   N=500
   x=np.random.normal(50,2,N)
   y=np.random.normal(0,1,N)
   z=np.random.normal(25,5,N)
   data=np.vstack((x,y,z))
   data=data.T
   
   out = tsneBH(data,showprogress=2)

   from scatter import scatter
   scatter(out[:,0],out[:,1], size=100)

   size = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
   labx=np.hstack((np.repeat('labx_1',50),np.repeat('labx_2',25),np.repeat('labx_3',25)))
   scatter(out[:,0],out[:,1], labx=labx, labxtype='unique', size=size, density=0, density_levels=25)

 SEE ALSO
   pca, UMAP
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : tsneBH.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Nov. 2017
#--------------------------------------------------------------------------

#%%
def tsneBH(data, components=2, perplexity=30, init='random', methodtype='barnes_hut', random_state=None, metric='euclidean', showprogress=2):
	#%% DECLARATIONS
    out =[]
    # Make dictionary to store Parameters
    Param = {}
    Param['components']   = components
    Param['perplexity']   = perplexity
    Param['init']         = init
    Param['methodtype']   = methodtype
    Param['random_state'] = random_state
    Param['metric']       = metric
    Param['showprogress'] = showprogress

    #%% Libraries
#    import sklearn.datasets
#    import sklearn.decomposition
    import sklearn.manifold
    
    #%% SET PARAMTERS FOR TSNE
    tsne = sklearn.manifold.TSNE(n_components=Param['components'], method=Param['methodtype'], init=Param['init'], random_state=Param['random_state'],verbose=Param['showprogress'])
    #pca  = sklearn.decomposition.TruncatedSVD(n_components=2)

    #%% Perform the reduction
#    out1 = tsne.get_params(deep=True)
    out = tsne.fit_transform(data)
    #embed_pca = pca.fit_transform(df)

    #%% END
    return(out)
