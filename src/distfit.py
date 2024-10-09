""" This function checks 89 different distributions and computes which one fits 
    best to your emperical distribution based on Mean Square error (MSE) estimates.

	[out_dist, out] = distfit(data, <optional>)

 INPUT:
   data:           Numpy array
                   

 OPTIONAL

   bins=           [Integer]: Bin size to make the estimation
                   [50]: (default)

   alpha=          Double : [0..1] Significance alpha
                   [None]: Default
   
   bound=          String: Set whether you want returned a P-value for the lower/upper bounds or both
                   'both': Both (default)
                   'up':   Upperbounds
                   'down': Lowerbounds

   distribution=   String: Set the distribution to use
                   'auto_small': A smaller set of distributions: [norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm] (default) 
                   'auto_full' : The full set of distributions
                   'norm'      : normal distribution
                   't'         : Students T distribution
                   etc

   title=          String Title of the figure
                   '' (default)

   showfig=        [Boolean] [0,1]: Show figure
                   [0]: No
                   [1]: Yes (Default)

   showprogress=   [Boolean] [0,1]
                   [0]: No (default)
                   [1]: Yes
                   [2]: Yes (More information)

 OUTPUT
	output

 INFO:
   https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python?lq=1
   https://stackoverflow.com/questions/7718034/maximum-likelihood-estimate-pseudocode

 #EXAMPLE
   %reset -f
   import sys, os, importlib
   
   print(os.getcwd())
   import distfit
   
   import pandas as pd
   import numpy as np

   # Generate data from normal distribution
   data=np.random.normal(5, 8, [1000,1000])
   data=np.random.normal(5, 8, 10000)
   #data=data.tolist()
   #data=pd.Series(data)

   # Find best fit distribution
   bins=50
   [fitparam, fitname] = distfit(data, bins=bins, distribution='auto_small', showfig=1, showprogress=1, width=8,  height=8)
   [fitparam, fitname] = distfit(data, bins=bins, distribution='auto_small', showfig=1,showprogress=2)
   [fitparam, fitname] = distfit(data, bins=bins, distribution='auto_small', alpha=0.05, showfig=1,showprogress=1)
   [fitparam, fitname] = distfit(data, bins=bins, distribution='norm', showfig=1)
   print("Estimated distribution: %s [mu:%f, std:%f]" %(fitname,fitparam[0],fitparam[1]))
   
 SEE ALSO: hypotesting
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : distfit.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Company     : Rijkswaterstaat
#--------------------------------------------------------------------------


#%%
def distfit(data, bins=50, distribution='auto_small', alpha=None, bound='both', title='', showfig=1, width=8,  height=8, showprogress=2): 
	#%% DECLARATIONS
    out      = []
    Param    = {}
    out_dist = {}
    Param['showprogress'] = showprogress
    Param['bins']         = bins
    Param['showfig']      = showfig
    Param['distribution'] = distribution
    Param['alpha']        = alpha
    Param['bound']        = bound
    Param['title']        = title
    Param['width']        = width
    Param['height']       = height

    #%% Libraries
    import warnings
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    import matplotlib.pyplot as plt
    from showprogress import showprogress
    from hist import hist

    #%%
    if len(data)==0:
        return (out_dist, out)
    #end
    
    #%% Convert pandas to numpy
    if 'pandas' in str(type(data)):
        data = data.values
    #end
    if str(data.dtype)=='O':
        data=data.astype(float)
    #end

    #%% Make sure its a vector
    data = data.ravel()
    
    #%% THE REAL THING
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    [yObs, x] = np.histogram(data, bins=Param['bins'], density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    #plt.plot(x,yObs)

    #%% Confidence intervals
    #TODO

    #%% Distributions to check
    if Param['distribution']=='auto_full':
        DISTRIBUTIONS = [st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.laplace,st.levy,st.levy_l,st.levy_stable,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,
            st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]
    elif Param['distribution']=='auto_small':
        DISTRIBUTIONS = [st.norm, st.expon, st.pareto, st.dweibull, st.t, st.genextreme, st.gamma, st.lognorm]
    else:
        # Connect object with variable to be used as a function again.
        DISTRIBUTIONS  = [getattr(st, Param['distribution'])]
    #end

    #%% Best holders
#    best_dist   = st.norm
#    best_params = (0.0, 1.0)
    out_dist['distribution'] =  st.norm
    out_dist['params']       = (0.0, 1.0)
    best_sse    = np.inf
    out         = pd.DataFrame(index=range(0,len(DISTRIBUTIONS)), columns=['Distribution','SSE','LLE','loc','scale','arg'])
    #out        = np.zeros(len(DISTRIBUTIONS),dtype=float)
    
    #%% Estimate distribution parameters from data
    i=0
    for distribution in DISTRIBUTIONS:
        logLik=0
#        tic()
#        if Param['showprogress']==1:
#            print("Checking for [%s]" %distribution.name)
        
        # Try to fit the distribution. However this can result in an error so therefore you need to try-catch
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg   = params[:-2]
                loc   = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                # Compute SSE
                sse = np.sum(np.power(yObs - pdf, 2.0))
                # Compute Maximum likelhood
#                x = np.linspace(0, 100, num=100)
#                yObs = 5 + 2.4*x + np.random.normal(0, 4, len(data))
#                b0 = params[0]
#                b1 = params[1]
#                nu = params[2]
#                yPred = b0 + b1*x
#                logLik = -np.log( np.prod(pdf(x, mu=yPred, nu=nu)))


#                logLik=0
                # Calculate the negative log-likelihood as the negative sum of the log of a normal
                # PDF where the observed values are normally distributed around the mean (yPred)
                # with a standard deviation of sd
                # Calculate the predicted values from the initial parameter guesses
#                yPred = params[0] + params[1]*x
                try:
                    logLik = -np.sum( distribution.logpdf(yObs, loc=loc, scale=scale) )
                except Exception:
                    logLik = float('NaN')
                    pass
#                if len(params)>2:
#                    logLik = -np.sum( distribution.logpdf(yObs, arg=arg, loc=loc, scale=scale) )
#                else:
#                    logLik = -np.sum( distribution.logpdf(yObs, loc=loc, scale=scale) )
                
#                # Store results
                out.values[i,0] = distribution.name
                out.values[i,1] = sse
                out.values[i,2] = logLik
                out.values[i,3] = loc
                out.values[i,4] = scale
                out.values[i,5] = arg
                
                # if axis pass in add to plot
#                try:
#                    if ax:
#                        pd.Series(pdf, x).plot(ax=ax)
#                except Exception:
#                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
#                    best_dist   = distribution
                    #best_params = params
                    best_sse                 = sse
                    out_dist['name']         = distribution.name
                    out_dist['distribution'] = distribution
                    out_dist['params']       = params
                    out_dist['sse']          = sse
                    out_dist['loc']          = loc
                    out_dist['scale']        = scale
                    out_dist['arg']          = arg
                #end

            # Showprogress bar
            if Param['showprogress']==2:
                print("Checking for [%s] [SSE:%f] [logLik:%f]" %(distribution.name,sse,logLik))
                #showprogress(i,len(DISTRIBUTIONS))
            #end
            
            i=i+1
        except Exception:
            pass
        #end
        
    #%% Sort the output
    out = out.sort_values('SSE')
    # Reset index too
    out.index = range(0,len(DISTRIBUTIONS))
    
    #%% Generate distributions's Propbability Distribution Function
    best_dist      = out_dist['distribution']
    best_fit_name  = out_dist['name']
    best_fit_param = out_dist['params']

    # Separate parts of parameters
    arg   = out_dist['params'][:-2]
    loc   = out_dist['params'][-2]
    scale = out_dist['params'][-1]
    size  = len(data)

    # Determine %CI
    dist   = getattr(st, best_fit_name)
    CIIup = None
    CIIdown = None
    if Param['alpha']!=None:
        if Param['bound']=='up' or Param['bound']=='both':
            CIIdown = dist.ppf(1-Param['alpha'], *arg, loc=loc, scale=scale) if arg else dist.ppf(1-Param['alpha'], loc=loc, scale=scale)
        #end
        if Param['bound']=='down' or Param['bound']=='both':
            CIIup = dist.ppf(Param['alpha'], *arg, loc=loc, scale=scale) if arg else dist.ppf(Param['alpha'], loc=loc, scale=scale)
        #end
    #end
    
    # Store
#    out_dist['CII_min_'+str(Param['alpha'])]=CIIup
#    out_dist['CII_max_'+str(Param['alpha'])]=CIIdown
    out_dist['CII_min_alpha']=CIIup
    out_dist['CII_max_alpha']=CIIdown

    #%% Make figure
    out_dist['ax']=None
    if Param['showfig']:
        # Plot line
        getmin = dist.ppf(0.0000001, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.0000001, loc=loc, scale=scale)
        getmax = dist.ppf(0.9999999, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.9999999, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x   = np.linspace(getmin, getmax, size)
        y   = dist.pdf(x, loc=loc, scale=scale, *arg)

        # plt.figure(figsize=(6,4))
        [fig, ax]=hist(data,bins=Param['bins'],xlabel='Values',ylabel='Frequency', grid=True, normed=1, showprogress=Param['showprogress']>=1, width=Param['width'],height=Param['height'])
        plt.plot(x, y, 'b-', linewidth=2)
        legendname=[best_fit_name,'Emperical distribution']
    
        # Plot vertical line To stress the cut-off point
        if Param['alpha']!=None:
            if Param['bound']=='down' or Param['bound']=='both':
                ax.axvline(x=CIIup, ymin=0, ymax=1, linewidth=2, color='r', linestyle='dashed')
                legendname=[best_fit_name,'CII low '+'('+str(Param['alpha'])+')', 'Emperical distribution']
            #end
            if Param['bound']=='up' or Param['bound']=='both':
                ax.axvline(x=CIIdown, ymin=0, ymax=1, linewidth=2, color='r', linestyle='dashed')
                legendname=[best_fit_name,'CII high '+'('+str(Param['alpha'])+')', 'Emperical distribution']
            #end
            if Param['bound']=='both':
                legendname=[best_fit_name,'CII low '+'('+str(Param['alpha'])+')','CII high '+'('+str(Param['alpha'])+')','Emperical distribution']
            #end
        #end
        
        plt.legend(legendname)
        # Make text for plot
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str   = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_param)])
        dist_str    = '{}({})'.format(best_fit_name, param_str)
        ax.set_title(Param['title']+'\nBest fit distribution\n' + dist_str)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')    

        #Store axis information
        out_dist['ax']=ax
    #end
    
    #%%
    if Param['showprogress']>=1:
        print("Estimated distribution: %s [loc:%f, scale:%f]" %(out_dist['name'],out_dist['params'][-2],out_dist['params'][-1]))
    #end
        
    #%% END
    return (out_dist, out)
    #return (out_dist, best_dist.name, best_params, out)
