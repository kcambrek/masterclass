""" This function provides various methods for cluster validation

	A= clusteval(data, <optional>)

   data:         datamatrix
                 rows    = features
                 colums  = samples
 OPTIONAL

 method=          String: Method type for cluster validation
                 'silhouette' (default)
                 'dbindex'
                 'derivative'

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
   import clusteval
   

   from sklearn.datasets.samples_generator import make_blobs
   [data, labels_true] = make_blobs(n_samples=750, centers=5, n_features=10)
   out1= clusteval(data, method='silhouette')
   out2= clusteval(data, method='dbindex')
   out3= clusteval(data, method='derivative')



 SEE ALSO
   silhouette, silhouette_plot, dbindex, elbowclust
   
print(__doc__)
"""

#--------------------------------------------------------------------------
# Name        : clusteval.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------

#from matplotlib.pyplot import plot

#%% Libraries
from dbindex import dbindex
from silhouette import silhouette
from elbowclust import elbowclust

#%%
def clusteval(X, method='silhouette', metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, width=15, height=8, showfig=1, verbose=1):
	#%% DECLARATIONS
    out ={}
    
    # Make dictionary to store Parameters
    Param = {}
    Param['method']      = method
    Param['verbose']     = verbose
    Param['metric']      = metric
    Param['linkage']     = linkage
    Param['minclusters'] = minclusters
    Param['maxclusters'] = maxclusters
    Param['showfig']     = showfig
    Param['height']      = height
    Param['width']       = width
        
    #%% Cluster hierarchical using on metric/linkage
    from scipy.cluster.hierarchy import linkage
    Z=linkage(X, method=Param['linkage'], metric=Param['metric'])
    
    #%% THE REAL THING
    if Param['method']=='silhouette':
        out=silhouette(X, Z=Z, minclusters=Param['minclusters'] , maxclusters=Param['maxclusters'] , width=Param['width'], height=Param['height'], showfig=Param['showfig'], verbose=Param['verbose'])


    if Param['method']=='dbindex':
        out=dbindex(X, Z=Z, minclusters=Param['minclusters'] , maxclusters=Param['maxclusters'] , width=Param['width'], height=Param['height'], showfig=Param['showfig'], verbose=Param['verbose'])


    if Param['method']=='derivative':
        out=elbowclust(X, Z=Z, minclusters=Param['minclusters'] , maxclusters=Param['maxclusters'] , width=Param['width'], height=Param['height'], showfig=Param['showfig'], verbose=Param['verbose'])


    return(out)
