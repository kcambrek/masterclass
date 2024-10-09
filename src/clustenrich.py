""" This function computes significance or enrichment for the clusterlabels with the catagorical/numerical/continues values

	A= clustenrich(data, labx, <optional>)

 INPUT:
   data:           Pandas DataFrame for which the columns should match with the number of labx
                   rows: samples
                   cols: features
       
   labx:           vector of labels. Can be of type numeric or string. In a two-class model, always use [1] as the positive class!
                   [1,1,1,3,4,5,2,1,4,...]
                   ['1','1','1','3',4','5','2','1','4',...]
                   [1,1,1,0,0,0,...] # 1 is the positve class. Otherwise the first one in alphabetic order will be taken

 OPTIONAL

   alpha=          Double : [0..1] Significance alpha
                   [0.05]: Default
   
   dtype:          Type of column, either being nume
                   'pandas'
                   ['1','1','1','3',4','5','2','1','4',...]

   multtest=       [String]:
                   'none'           : No multiple Test (default)
                   'bonferroni'     : one-step correction
                   'sidak'          : one-step correction
                   'holm-sidak'     : step down method using Sidak adjustments
                   'holm'           : step-down method using Bonferroni adjustments
                   'simes-hochberg' : step-up method  (independent)
                   'hommel'         : closed method based on Simes tests (non-negative)
                   'fdr_bh'         : Benjamini/Hochberg  (non-negative)
                   'fdr_by'         : Benjamini/Yekutieli (negative)
                   'fdr_tsbh'       : two stage fdr correction (non-negative)
                   'fdr_tsbky'      : two stage fdr correction (non-negative)

   verbose=        Boolean [0,1]
                   [0]: No (default)
                   [1]: Yes

 OUTPUT
	output

 DESCRIPTION
   The cluster labels (labx) should be a vector with the same number of samples as for the input data.
   For each column in data, significance is assessed for the labels in a two-class approach (labx==1 versus labx~=1 etc)
   Significance is assessed one tailed. Thus whether there is significant enrichment for the label with a specific column. NOT the absence of it.
   Fisher-exact test is used for catagorical values
   Wilcoxen rank-sum for continues values
   
   This results in a probability: P(group in cluster | known clusters)
   
 EXAMPLE
   %reset -f
   import sys, os, importlib
   print(os.getcwd())
   import clustenrich
   

   from clusteval import clusteval
   import pandas as pd
   df   = pd.read_csv('cancer_xy.csv')
   labx = clusteval(df[['x','y']].values, showfig=0)['labx']
   data = df[['age','sex','survival_months','labx']]

   #labx=data.labx
   #labx=np.array(labx==1,dtype=int)
   A = clustenrich(data, labx==1, alpha=0.01, verbose=1, dtype='pandas')
   A = clustenrich(data, labx, alpha=0.01, verbose=1, dtype=['num','cat','num','cat'])

 SEE ALSO
   
print(__doc__)
"""

#--------------------------------------------------------------------------
# Name        : clustenrich.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------

#from matplotlib.pyplot import plot

#%%
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, ranksums
import statsmodels.stats.multitest as multitest

#%%
def clustenrich(data, labx, alpha=0.05, multtest='fdr_bh', dtype='pandas', verbose=1):
	#%% DECLARATIONS
    out =[]
    # Make dictionary to store Parameters
    Param = {}
    Param['verbose']  = verbose
    Param['alpha']    = alpha
    Param['multtest'] = multtest
    Param['dtype']    = dtype

    #%% Check pandas!!
    if not 'pandas' in str(type(data)):
        print('INPUT must be pandas-dataframe!')
        return(out)
    #end

    #%% Check labx with number of samples
    data.reset_index(drop=True, inplace=True)
    if data.shape[0]!=len(labx):
        print('data rows and labels are not equal! <return>')
        # return(out)
    #end

    if 'pandas' in str(type(labx)):
        labx=labx.values
    #end

    #%% Setup categorical/numerical
    if 'str' in str(type(Param['dtype'])):
        Param['dtype']=['']*len(data.columns)
        print('>Auto setting categorical/numerical for the columns..')
        for i in range(0,len(data.columns)):
            if 'float' in str(data.dtypes[i]):
                Param['dtype'][i] = 'num'
            elif 'int' in str(data.dtypes[i]):
                Param['dtype'][i] = 'cat'
            elif 'str' in str(data.dtypes[i]):
                Param['dtype'][i] = 'cat'
            elif 'object' in str(data.dtypes[i]):
                Param['dtype'][i] = 'cat'
            else:
                Param['dtype'][i] = 'cat'
            #end
            print('   >%s: [%s]' %(data.columns[i], Param['dtype'][i]))
        #end
    elif len(Param['dtype'])!=len(data.columns):
        print('STOP! Number of columns=[%d] does not match number of input-types=[%d]' %(len(data.columns), len(Param['dtype'])))
        return(out)
    #end
    #%% Convert boolean to [0,1]
    if 'bool' in str(type(labx[0])):
        idxT=np.where(labx==True)[0]
        labx=np.zeros(len(labx),dtype=int)
        labx[idxT]=1
    #end
    
    #%% Get column-names
    cols     = data.columns
    sampleid = np.arange(0,data.shape[0])
    out      = pd.DataFrame()

    #%% Convert dtypes as good as possible
    for i in range(0,len(data.columns)):
        if  Param['dtype']=='num':
            data[cols[i]] = data[cols[i]].astype(float)
        elif Param['dtype']=='cat':
            data[cols[i]] = data[cols[i]].astype(str)
        #end
    #end

    #%% Run over all possible columns
    for i in range(0,len(cols)):
        if Param['verbose']: print('>Analyzing [%s][%s]' %(cols[i],Param['dtype'][i]))

        ############ Get data from field
        getdata=data[cols[i]].values

        ############# Remove NaN fields
        tmpdata   = getdata.astype(str)
        remidx    = np.where(tmpdata=='nan')[0]
        getdata   = np.delete(getdata,remidx)
        labxF     = np.delete(labx, remidx)
        sampleidF = np.delete(sampleid, remidx)
        #uidata    = np.unique(getdata)
        tmpout    = []

        ############# Check whether it is catagorical or numerica data
        #percUnique=len(uidata)/len(getdata)

        if Param['dtype'][i]=='cat': #'str' in str(type(getdata[0])) or percUnique>0.667:
            tmpout = prob_categorical(getdata, labxF, cols[i], 'categorical', sampleidF)
        else:
            tmpout = prob_numerical(getdata, labxF, cols[i], 'numerical')
        #end

        ########### Store in dataframe
        tmpout=pd.DataFrame(tmpout, columns=['labx','category','type','class_overlap_cat','class_tot','class_out','class_median','class_other_median', 'P','stats', 'datatype'])
        
        
        ###########  If there are two classes, take one as the other one is 1-P
        uilabx=tmpout.labx.unique()
        if len(uilabx)==2:
            idx=[0] # Take the first (default)
            if 'int' in str(type(labx[0])) and  np.any(uilabx==1):
                idx=np.where(tmpout.labx==1)[0]
            #end
            tmpout = tmpout.iloc[idx,:]
        #end
        
        ########### Multiple test
        if multtest!='none' and multtest!='':
            padj = multitest.multipletests(tmpout.P.values, alpha=Param['alpha'], method=Param['multtest'])
            tmpout['Padj'] = padj[1]
        else:
            tmpout['Padj']=tmpout.P.values
        #end
        
        ############ Store in list
        out = pd.concat((out,tmpout))
    #end

    #%% Filter only for significant ones
    out = out.iloc[np.where(out.Padj<=Param['alpha'])[0],:]
    out.sort_values(by=['labx','Padj'], ascending=True, inplace=True)
    out.reset_index(drop=True, inplace=True)

    # Return
    return(out)

#%% Compute Pvalue for categorical labels
def prob_numerical(getdata, labxF, coli, datatype):
    out    = []
    uilabx = np.unique(labxF)
    
    for k in range(0,len(uilabx)):
        # Get data class1/class2
        class1=getdata[labxF==uilabx[k]]
        class2=getdata[labxF!=uilabx[k]]

        # Only continue if there are more then 3 positive labels!
        if len(class1)>=3 and len(class2)>=3:
            # Wilcoxen ranksum test
            [stats, P] = ranksums(class1,class2)
            # Store in array
            out.append([uilabx[k], coli, None, None, None, None, np.median(class1), np.median(class2), P, stats, datatype])
        #end

    #end
    return(out)

#%% Compute Pvalue for categorical labels
def prob_categorical(getdata, labxF, coli, datatype, sampleidF):
#    coli=cols[i] 
#    datatype='categorical'

    # M: Population Size                       (10,000 ->total number of genes))
    # n: Number of Successes in Population     (2000 -> known in pathway)
    # N: Sample Size                           (300 -> over-expressed genes)
    # x: Number of Successes in Sample         (60  -> overlap n with N)
    # P = hypergeom.cdf(60, 10000, 2000, 300)
    # prb = hypergeom.cdf(x, M, n, N)
    # Which should result in a probability of p ~ 0.52 to draw 60 F-associated genes or more from 300 randomly selected genes in the list -- not really very significant at all!

    M      = len(getdata)
    uilabx = np.unique(labxF)
    uidata = np.unique(getdata)

    out = []
    # Check enrichment for each catagory of columns with (cluster) label
    for k in range(0,len(uilabx)):
        # Sample Size
        N=sampleidF[labxF==uilabx[k]]
        # Check enrichment for that particular catagory in the column with every possible (cluster) label
        for p in range(0,len(uidata)):
            # Number of Successes in Population
            n=sampleidF[np.where(getdata==uidata[p])[0]]
            # Number of Successes in Sample
            X = np.intersect1d(n,N)
            # Hypergeometric probability
            P = 1-hypergeom.cdf(len(X), M, len(n), len(N))
            #P = hypergeom.sf(len(X), M, len(n), len(N), loc=0)
            #print("%.3f vs %.3f" %(P1,P))
            # Store info
            out.append([uilabx[k], coli, uidata[p], len(X), len(N),  len(np.setdiff1d(sampleidF,N)),  None, None, P, None, datatype])
        #end
    #end
    
    return(out)