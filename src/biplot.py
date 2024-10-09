""" This function makes the biplot for PCA

	A= biplot(data, <optional>)

 INPUT:
   data:           datamatrix
                   rows    = features
                   colums  = samples
 OPTIONAL

   components=     Integer: Number of components for feature reduction.
                   [2]: (default)

   pcp=            Float: Take PCs with percentage explained variance>pcp
                   [0.95] (Number of componentens that cover the 95% explained variance)

   topn=           Integer: Show the top n loadings in the figure per principal component. The first PCs are demonstrated thus the unique features in 2x toploadings
                   [25] (default)
                   
   labx=           list of strings of length [x]
                   [] (default)
   
  features=        Numpy Vector of strings: Name of the features that represent the data features and loadings
                   [] (default)
                  
  savemem=         Boolean [0,1]
                   [0]: No (default)
                   [1]: Yes (the output of the PCA is directly the embedded space, and not all PCs. This will affect the explained-variance plot)

   height=         Integer:  Height of figure
                   [5]: (default)

   width=          Integer:  Width of figure
                   [5]: (default)

  showfig=         Integer [0,1, 2]
                   [0]: No
                   [1]: Plot explained variance
                   [2]: 2D biplot of 1st and 2nd PC
                   [3]: all of the above
                   
   showprogress=   Boolean [0,1]
                   [0]: No (default)
                   [1]: Yes

 OUTPUT
	output

 DESCRIPTION
   Show the loadings of the PCA
   https://plot.ly/ipython-notebooks/principal-component-analysis/
   https://sukhbinder.wordpress.com/2016/03/02/biplot-in-python-revisited/

 EXAMPLE
   %reset -f
   import sys, os, importlib
   
   print(os.getcwd())
   import biplot
   

   from sklearn.datasets import load_iris
   import numpy as np
   iris = load_iris()
   data=iris.data
   features=np.array(iris.feature_names)
   labx=np.array(iris.target)

   import pandas as pd
   labx = df.labx.values
   data = df[['x','y']].values
   
   
   A = biplot(data, components=2, labx=labx, features=features)
   A = biplot(data, components=0.99, labx=labx, features=features)
   A = biplot(data, labx=labx, features=features, topn=1)

 SEE ALSO
   PCA
   
print(__doc__)
"""

#--------------------------------------------------------------------------
# Name        : biplot.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------

#from matplotlib.pyplot import plot

#%%
def biplot(data, components=2, topn=25, labx=[], features=[], savemem=0, xlim=[], ylim=[], height=8, width=10, showfig=3, grid=True, showprogress=1):
	#%% DECLARATIONS
    out ={}
    # Make dictionary to store Parameters
    Param = {}
    Param['showprogress'] = showprogress
    Param['components']   = components
    Param['labx']         = labx
    Param['features']     = features
    Param['topn']         = topn
    Param['width']        = width
    Param['height']       = height
    Param['savemem']      = savemem
    Param['pcp']          = 0.95 # PCs that cover percentage explained variance
    Param['xlim']         = xlim
    Param['ylim']         = ylim
    Param['grid']         = grid
    Param['showfig']      = showfig
    
    #%% Checks
    if Param['topn']<1:
        print('Cannot run with [topn]<1')
        return
    #end

    #%% Libraries
    import numpy as np
    from sklearn.decomposition import PCA
    from scatter import scatter
    import matplotlib.pyplot as plt

    #%% Set featnames of not exists
    if Param['features']==[] or len(Param['features'])!=data.shape[1]:
        Param['features'] = np.arange(1,data.shape[1]+1).astype(str)
    #end
    
    if len(Param['labx'])!=data.shape[0]:
        #print("WARNING: labx length did not match! Expected=[%d], input=[%d]" %(data.shape[0], len(Param['labx'])))
        Param['labx']=[]
    #end

    #%% THE REAL THING
    if Param['savemem']==1 or Param['components']<1:
        pca=PCA(n_components=Param['components'])
    else:
        pca=PCA(n_components=data.shape[1])
    #end
    # pca=PCA(n_components=0.95)
    pca.fit(data)

    #%% project data into PC space
    # 0,1 denote PC1 and PC2; change values for other PCs
    loadings = pca.components_ # Ook wel de coeeficienten genoemd: coefs!
    PC       = pca.transform(data) # Ook wel SCORE genoemd
#    coefs  = pca.components_
#    score  = pca.transform(data)
#    latent = pca.explained_variance_
    
    #%% Set number of components based on PCs with input % explained variance
    if Param['components']<1:
        Param['pcp'] = Param['components']
        Param['components']=PC.shape[1]
    #end

    #%% Compute explained variance, top 95% variance
    #percentExplVar             = np.cumsum(latent) / sum(latent)
    percentExplVar             = pca.explained_variance_ratio_.cumsum()
    pcVar                      = np.min(np.where(percentExplVar>Param['pcp'])[0])+1  # Plus one because starts with 0
    out['expl_var_perc']       = percentExplVar
    out['PCs_95_perc_expl_nr'] = pcVar
#    out['PCs_95_perc_expl']    = PC[:,0:pcVar]
    out['PC']                  = PC[:,0:Param['components']]

    #%% Top scoring components
    # Top scoring for 1st component
    I1=np.argsort(np.abs(loadings[0,:]))
    I1=I1[::-1]
    # Top scoring for 2nd component
    I2=np.argsort(np.abs(loadings[1,:]))
    I2=I2[::-1]
    # Take only top loadings
    I1=I1[0:np.min([Param['topn'],len(I1)])]
    I2=I2[0:np.min([Param['topn'],len(I2)])]
    I = np.append(I1,I2)
    # Unique without sort:
    indices=np.unique(I,return_index=True)[1]
    I = [I[index] for index in sorted(indices)]
    
    out['topFeat']  = Param['features'][I]
    out['loadings'] = loadings
    # score[I[0:np.min([Param['topn'],len(I)])],0:Param['components']]

    #%% Show explained variance plot
    if Param['showfig']==1 or Param['showfig']==3:
        [fig,ax]=plt.subplots(figsize=(Param['width'],Param['height']),facecolor='w', edgecolor='k')
        plt.plot(np.append(0,percentExplVar),'o-', color='k')
        plt.ylabel('Percentage explained variance')
        plt.xlabel('Principle Components')
        plt.xticks(np.arange(0,len(percentExplVar)+1))
        plt.ylim([0,1])
        titletxt='Cumulative explained variance\nMinimum components to cover the [' + str(Param['pcp']) + '] explained variance, PC=['+ str(pcVar)+  ']'
        plt.title(titletxt)
        plt.grid(Param['grid'])
        # Plot vertical line To stress the cut-off point
    #    ax.axvline(x=eps[idx], ymin=0, ymax=sillclust[idx], linewidth=2, color='r')
        ax.axhline(y=Param['pcp'], xmin=0, xmax=1, linewidth=0.8, color='r')
        plt.style.use('ggplot')
        plt.bar(np.arange(0,len(percentExplVar)+1),np.append(0,pca.explained_variance_ratio_),color='#3182bd', alpha=0.8)
        plt.show()
        plt.draw()
    #end

    #%% Scatter samples in 2D
    if Param['showfig']==2 or Param['showfig']==3:
        [fig,ax]=plt.subplots(figsize=(Param['width'],Param['height']),facecolor='w', edgecolor='k')
        plt.style.use('ggplot')
#        [fig,ax]=plt.subplots(figsize=(Param['width'],Param['height']),facecolor='w', edgecolor='k')
        xs      = PC[:,0]
        ys      = PC[:,1]
        xlabel  = 'PC1 ('+ str(pca.explained_variance_ratio_[0]*100)[0:4] + '% expl.var)'
        ylabel  = 'PC2 ('+ str(pca.explained_variance_ratio_[1]*100)[0:4] + '% expl.var)'
        title   ='Biplot of PC1 vs PC2.\nNumber of components to cover the [' + str(Param['pcp']) + '] explained variance, PC=['+ str(pcVar)+  ']'
        scatter(xs,ys, labx=Param['labx'], labx_type='unique', size=100, xlabel=xlabel, ylabel=ylabel, width=Param['width'], height=Param['height'], title=title, xlim=Param['xlim'], ylim=Param['ylim'], newfig=0, ax=ax)
        
        #% Gather top N loadings
        #    xvector = loadings[0][0:np.clip(Param['topn'],0,len(loadings[0]))]
        #    yvector = loadings[1][0:np.clip(Param['topn'],0,len(loadings[1]))]
        xvector = loadings[0,I]
        yvector = loadings[1,I]

        # Plot and scale values for arrows and text
        scalex = 1.0/(loadings[0,:].max() - loadings[0,:].min())
        scaley = 1.0/(loadings[1,:].max() - loadings[1,:].min())
        
#        scaleToFig=np.minimum(np.abs(ys).max(), np.abs(xs).max())/2
        
#        scalex = (np.abs(xs).max() - np.abs(xs).min())
#        scaley = (np.abs(ys).max()- np.abs(ys).min())
        for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
            newx=xvector[i]*scalex
            newy=yvector[i]*scaley
            
            figscaling=np.abs([np.abs(xs).max()/newx, np.abs(ys).max()/newy])
            figscaling=figscaling.min()
            newx=newx*figscaling*0.5
            newy=newy*figscaling*0.5
            
            ax.arrow(0, 0, newx, newy, color='r', width=0.005, head_width=0.05, alpha=0.6)
            # plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys), color='r', width=0.0005, head_width=0.0025)
            ax.text(newx*1.25, newy*1.25, Param['features'][i], color='black', ha='center', va='center')
        #end
#        plt.style.use('ggplot')
    
#        plt.show()
#        plt.draw()
    #end

    #%% END
    return(out)
