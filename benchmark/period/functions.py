#!/usr/bin/env python3
import os
import numpy as np
from astropy.table import Table
from astropy.time import Time
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import pandas as pd

# Complementary filters to compute the colour
# RRL
# comp_filt = {'U':'B','B':'V','V':'B','R':'V','I':'V'}
# comp_col = {'U':'U-B','B':'B-V','V':'B-V','R':'V-R','I':'V-I'}
# sign = {'U':+1,'B':+1,'V':-1,'R':-1,'I':-1}
# Cep
comp_filt = {'U':'B','B':'V','V':'R','R':'V','I':'V'}
comp_col = {'U':'U-B','B':'B-V','V':'V-R','R':'V-R','I':'V-I'}
sign = {'U':+1,'B':+1,'V':+1,'R':-1,'I':-1}

def prepareTable(file, hdu, index=None):
  ###################################################################################
  # Function to read and prepare the fits tables computed with gnuastro as pandas'
  # dataframes for operating with them.
  #     Input arguments:  - file = fits file to be read.
  #                       - hdu = hdu where the data must be read from.
  #                       - index = None by default. Sets the dataframe's index from
  #                                an internal columns from the fits file if defined.
  #      
  #     Output arguments: - data: panda's dataframe with the data.
  ###################################################################################

  # Read the table from file, known its header, and the column used as index.
  data = Table.read(file, hdu=hdu).to_pandas()
  if index != None:
    data.set_index(index,inplace=True)
  # Check for duplicate rows and erase them if they exist.
  data.drop_duplicates(inplace=True)
  
  # Corrects format for columns containing strings that come as byte type.
  if data.select_dtypes(object).stack().shape[0] != 0:
    str_typ = data.select_dtypes(object).stack().str.decode("utf-8").unstack()
    for col in str_typ:
      data[col] = str_typ[col].str.rstrip()
  
  data["DATE-OBS"] = Time(data["DATE-OBS"].values.astype(str)).mjd
  
  return data

def quantities(Table, crit, filt, dat=False):
  ###################################################################################
  # Function to extract some columns from the catalog and see their evolution with 
  # time if tevol=True.
  #     Input arguments:  - Table = table containing the catalog from where to 
  #                                 extract the quantities.
  #                       - crit = conditions that the quantities must fulfil.
  #                       - filt = observed filter.
  #                       - **dat = False by default. Indicates if a variable 
  #                                 containing the full dataframe with the selected
  #                                 conditions wants to be settled or not.
  #      
  #     Output arguments: - airmass, colour, standard magnitude (mstd), experimental
  #                         magnitude (mins) and difference mstd-mins.
  ###################################################################################
  # TODO: añadir aquí que suelte un dataframe por cada filtro y criterio para poder hacer las medianas.
  data = Table[crit]
  
  mstd = data["MAG_"+filt]
  mins = data.MAG_AUTO_NORM
  mdiff = mstd-mins
  median, q1, q3, lowl, upl = interquartil(mstd,mins)
  data = data[(mdiff<upl)&(mdiff>lowl)]
  
  mstd = data["MAG_"+filt]
  mins = data.MAG_AUTO_NORM
  mdiff = mstd-mins
  
  airmass = data.AIRMASS
  colour = data[comp_col[filt]]
#  xx = stdTable[crit].X_IMAGE
#  yy = stdTable[crit].Y_IMAGE
#  mins = data.MAG_AUTO+2.5*np.log10(data.EXPTIME)
#  pd.options.mode.chained_assignment = None
#  data.loc["MAG_DIFF",:] = mdiff
#  pd.options.mode.chained_assignment = 'warn'
  dates = data["DATE-OBS"]
  err = data.MAGERR_AUTO
  
  if dat==True:
    return data, airmass, colour, mstd, mins, mdiff, dates, err
  elif dat==False:
    return airmass, colour, mstd, mins, mdiff, dates, err
  else: 
    raise NameError('tevol must be boolean')

def timejump(Table):
  # TODO: comentar
  
  dates = np.unique(Table["DATE-OBS"])
  dates.sort()
  step = dates[1:] - dates[:-1]
  
  jumps = dates[np.where(step>np.median(step)*10)]
  
  return jumps

def criteria(Table, mmin, mmax, filt, threshold, tevol=False, dates=0, pos=0):
  ###################################################################################
  # Function set the criteria to extract the data from the catalog: standard
  # magnitude between max and min, filter, a maximun magnitude of the complementary 
  # filter used to compute the colour to avoid bad stars with m=99, and observing 
  # time if tevol=True.
  #     Input arguments:  - Table = table containing the catalog.
  #                       - mmin = minimum std magnitude to consider.
  #                       - mmax = maximum std magnitude to consider.
  #                       - filt = observed filter.
  #                       - tevol = False by default. If True, computes the 
  #                                 coefficients for each image in the night to 
  #                                 see their evolution.
  #                       - **dates = array of dates to consider. 0 by default.
  #                       - **post = position in the dates array. inf by default.
  #      
  #     Output arguments: - crit: pandas boolean array that matches conditions.
  ###################################################################################

  if tevol == False:
    crit = (Table["MAG_"+filt]<mmax[filt])&(Table["MAG_"+filt]>mmin[filt])\
            &(Table.INSFILTE==filt)&(Table["MAG_"+comp_filt[filt]]<mmax[comp_filt[filt]])\
            &(Table["MAG_"+comp_filt[filt]]>mmin[comp_filt[filt]])\
            &((Table["MAGERR_AUTO"]/Table["MAG_AUTO"])<threshold[filt])
  elif tevol == True:
    dates = np.append(np.append(0,dates),np.inf)
    crit = (Table["MAG_"+filt]<mmax[filt])&(Table["MAG_"+filt]>mmin[filt])      \
            &(Table["DATE-OBS"]>=dates[pos])&(Table["DATE-OBS"]<=dates[pos+1])  \
            &(Table.INSFILTE==filt)&(Table["MAG_"+comp_filt[filt]]<mmax[comp_filt[filt]])
               
  else: 
    raise NameError('tevol must be boolean')
  
  return crit

def fitting(Table, mmin, mmax, filt, guess, tevol=False, crit=None, err=False):
  ###################################################################################
  #  Function to fit the experimental magnitude and find the photometric coefficients
  #  of a certain observed filter.
  #     Input arguments:  - Table = table containing the catalog.
  #                       - mmin = minimum std magnitude to consider in the fit.
  #                       - mmax = maximum std magnitude to consider in the fit.
  #                       - filt = observed filter.
  #                       - guess = initial guess for the coefficients of the fit.
  #                       - tevol = False by default. If True, computes the 
  #                                 coefficients for each image in the night to 
  #                                 see their evolution.
  #                       - crit: None by default. Criteria to extract the data from 
  #                               the table. If the default value is maintained, a
  #                               standard criteria will be computed.
  #                       - err: False by default. Consider errors in the fit or not.
  #      
  #     Output arguments: - coeff: coefficients of the fit, namely zeropoint, 
  #                                 extinction and colour coefficients.
  #                       - cov: covariance matrix of the fit.
  #                       - **fit: values of the function with the computed 
  #                                coefficients and values of airmass and colour. 
  #                                Only if teval = False.
  ###################################################################################
  
  if tevol == True:
   
    jumps = timejump(Table)
     
    #Define empty matrixes to write coefficients and covariance matrix for each date.
    coeffs = np.zeros((len(jumps)+1,3))
    covs = np.zeros((len(jumps)+1,3,3))
    
    # Loop for each date.
    for ii in range(len(jumps)+1):

      # Set the criteria to extract the data from the catalog.
      crit = criteria(Table, mmin, mmax, filt, tevol, jumps, ii)
      
      # Define the variables for the fit.
      airmass, colour, mstd, mins, mdiff, date, errm = quantities(Table, crit, filt)
      
      # Optimize coefficients and write them into the bigger variable containing all
      # of them.
      if err == False:
        coeff, cov = optimization.curve_fit(func, (airmass,colour), mdiff, guess)
      else:
        coeff, cov = optimization.curve_fit(func, (airmass,colour), mdiff, guess,      \
                                            sigma=errm)
      coeffs[ii,:] = coeff
      covs[ii] = cov
        
    return coeffs, covs, np.mean(date)
  
  elif tevol == False:
    
    if type(crit) == type(None):
      crit = criteria(Table, mmin, mmax, filt)

    airmass, colour, mstd, mins, mdiff, date, errm = quantities(Table, crit, filt)

    if err == False:
      coeff, cov = optimization.curve_fit(func, (airmass,colour), mdiff, guess)
    else:
      coeff,cov = optimization.curve_fit(func, (airmass,colour), mdiff, guess,       \
                                         sigma=errm)
    fit = func((airmass, colour), coeff[0], coeff[1], coeff[2])
    
    return coeff, cov, fit
    
  else: raise NameError('tevol must be boolean')


def func(var,zero,extinction,colourterm):
  ###################################################################################
  #  Function which is optimized and that defines the relation among magnitudes, 
  #  airmass and colour. 
  #     Input arguments:  - var = (airmass, colour).
  #                       - zero = zeropoint of m_std - m_ins.
  #                       - extinction = extinction coefficient.
  #                       - colourterm = colour coefficient.
  #      
  #     Output arguments: - relation between variables.
  ###################################################################################
  
  airmass,colour = var
  return zero + extinction*airmass + colourterm*colour

def plotfit(ax, var1, var2, xname, yname, label=None, title=None, col='blue', err1=0, err2=0):
  ###################################################################################
  #  Function to plot the different variables used in the fit, and their errors.
  #     Input arguments:  - ax = axis where the variables will be plotted.
  #                       - var1 = x variable.
  #                       - var2 = y variable.
  #                       - xname = label for the x axis.
  #                       - yname = label for the y axis.
  #                       - title = title for the plot.
  #                       - **err1: errors in x variable.
  #                       - **err2: errors in y variable.
  #      
  #     Output arguments: plots.
  ###################################################################################
  
  if xname == "Date":
    datesnum = Time(var1.values.astype(str)).jd
    if label != None:
      ax.plot(datesnum, var2, '.', color=col, label=label, ms=2)
    else:
      ax.plot(datesnum, var2, '.', color=col, ms=2)
    ax.set_xticklabels(var1,rotation=90)
  else:
    if label != None:
      ax.plot(var1, var2, '.', color=col, label=label, ms=2)
    else:
      ax.plot(var1, var2, '.', color=col, ms=2)
      
  ax.set_xlabel(xname)
  ax.set_ylabel(yname)
  if title != None:
    ax.set_title(title)
#  ax.set_aspect('equal')
  
  return None

def interquartil(mdiff,fit):
  ###################################################################################
  #  Function to get the median, the first and the third quartiles, and lower and 
  #  upper limits to define outlyers.
  #     Input arguments:  - mdiff = m_std - m_ins.
  #                       - fit = fit obtained using the coefficients.
  #      
  #     Output arguments: - median: median of mdiff-fit distribution.
  #                       - q1: first quartile of the mdiff-fit distribution.
  #                       - q3: third quartile of the mdiff-fit distribution.
  #                       - lowl: lower limit for considering outlyers, defined as 
  #                               first quartil - 1.5*interquartil range.
  #                       - upl: lower limit for considering outlyers, defined as 
  #                              third quartil + 1.5*interquartil range.
  ###################################################################################
  
  median = np.median(mdiff-fit)
#  std = np.std(mdiff-fit)
  q1 = np.quantile(mdiff-fit,0.25)
  q3 = np.quantile(mdiff-fit,0.75)
  intq = q3-q1
  
  lowl = q1-1.5*intq
  upl = q3+1.5*intq

  return median, q1, q3, lowl, upl
  
def plothist(ax, bins, mdiff, fit, title):
  ###################################################################################
  #  Function to plot an histogram of the difference among the fit and mdiff.
  #     Input arguments:  - ax = axis where the variables will be plotted.
  #                       - bins = number of bins for the histogram.
  #                       - mdiff = m_std - m_ins.
  #                       - fit = fit obtained using the coefficients.
  #                       - title = title for the plot.
  #      
  #     Output arguments: - plots.
  ###################################################################################

  median, q1, q3, lowl, upl = interquartil(mdiff,fit)
  
  ax.hist(mdiff-fit, bins=30, color='b', edgecolor='k', linewidth=0.5)
  ax.set_xlabel(r'$m_{std}-m_{ins}-fit$')
  ax.set_ylabel("Frequency")
  ax.axvline(median, color='r', label=r'$Distribution\,median$')
  ax.axvline(q1, color='r', ls=(0, (5, 5)), label="First quartile, q1 = %.3f" % q1)
  ax.axvline(q3, color='r', ls=(0, (5, 5)), label="Third quartile, q3 = %.3f" % q3)
  ax.axvline(lowl, color='r', ls=(0, (1, 5)), label=r'$q1-1.5IQR=%.3f$' % lowl)
  ax.axvline(upl, color='r', ls=(0, (1, 5)), label=r'$q3+1.5IQR=%.3f$' % upl)
#  ax.vlines([median-std, median+std], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],  \
#            color="r", ls=(0, (5, 5)), label=r'$\mu\,\pm\,2\sigma')
  ax.set_title(title)
  ax.legend()
            
  return median, q1, q3, lowl, upl

def minmaxmag(Table,mmin,mmax,filt,threshold):
  mins = Table.loc[(Table["INSFILTE"]==filt)&(Table["MAG_AUTO"]<90)&(Table["MAG_"+filt]<90),"MAG_AUTO"]
  mstd = Table.loc[(Table["INSFILTE"]==filt)&(Table["MAG_AUTO"]<90)&(Table["MAG_"+filt]<90),"MAG_"+filt]
  
  errstd = Table.loc[(Table["INSFILTE"]==filt)&(Table["MAG_AUTO"]<90)&(Table["MAG_"+filt]<90),"ERRMAG_"+filt]
  err = Table.loc[(Table["INSFILTE"]==filt)&(Table["MAG_AUTO"]<90)&(Table["MAG_"+filt]<90),"MAGERR_AUTO"]
  
  frq, bins = np.histogram(mstd.loc[(err/mins)<threshold],bins=50)
#   mmax[filt] = np.argwhere(frq>=2).max()
  mmax[filt] = 0
  meanfrq = 5 # np.mean(frq[:5])
  while meanfrq >= 1:
    subfrq = frq[mmax[filt]:][np.where(frq[mmax[filt]:]!=frq[mmax[filt]:].max())]
    secmax = subfrq.max()
    posmax = np.where(frq[mmax[filt]:]==secmax)[0][-1]# np.where(frq>1.5*np.mean(frq))[0][-1] # np.argmax(frq[mmax[filt]:])
    if posmax>=(len(frq)-2):
      posmax = np.where(frq[mmax[filt]:]==secmax)[0][-2]
    
    postwo = np.where((np.append(frq,0)[mmax[filt]+posmax:]>=2)==False)[0][0]
    mmax[filt] = mmax[filt] + posmax + postwo
#     mmax[filt] = np.argmax(frq) + np.where((np.append(frq,0)[np.argmax(frq):]>=2)==False)[0][0]
    if len(frq[mmax[filt]:])<5:
      break
    meanfrq = np.mean(frq[mmax[filt]:mmax[filt]+5])
#     print(meanfrq,mmax[filt])
  
  mmax[filt] = bins[mmax[filt]]
  
  median, q1, q3, lowl, upl = interquartil(mstd,mins)
  mmin[filt] = min(mstd[((mstd-mins)<upl)&((mstd-mins)>lowl)])

  # mmax[filt] = max(mstd[((mstd-mins)<upl)&((mstd-mins)>lowl)])
  # mmax[filt] = np.quantile(mstd[((mstd-mins)<upl)&((mstd-mins)>lowl)],0.1) #(mmax[filt]+mmin[filt])/2
  
  return mins, mstd, errstd, err, mmin, mmax
  
def photfit(Table, refstar, ax, ii, jj, kk, photometry):

  airmass = Table[Table.ID==refstar].AIRMASS
  
  if photometry=="AUTO":
    mag = Table[Table.ID==refstar].MAG_AUTO_NORM
    magerr = Table[Table.ID==refstar].MAGERR_AUTO
  elif photometry=="APER":
    mag = Table[Table.ID==refstar].MAG_APER_NORM
    magerr = Table[Table.ID==refstar].MAGERR_APER
    
  plotfit(ax, airmass, mag, "Airmass","Magnitude", \
          label="Star #"+str(refstar)+" in "+ii+" field", col=kk)
  fit, cov = np.polyfit(airmass, mag, 1, w=1/magerr, cov=True)     
  R2 = 1 - np.sum((mag-np.poly1d(fit)(airmass))**2)/np.sum((mag-np.mean(mag))**2)  
  ax.plot([np.min(airmass), np.max(airmass)],[fit[1]+fit[0]*np.min(airmass),        \
           fit[1]+fit[0]*np.max(airmass)], 'r', lw=0.2, \
           label=(r'$R^{2}=%.4f\,\,m=%.3f$' % (R2,fit[0])))
  ax.errorbar(airmass, mag, yerr=magerr, fmt='none', ecolor='k')#, capsize=5)
#  ax.legend()
           
  return fit, R2

#from astroML.utils.decorators import pickle_results
#from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
#from astroML.datasets import fetch_LINEAR_sample

#def compute_best_frequencies(data, datestr, magstr, magerrstr, n_eval=10000, n_retry=5, generalized=True):
#
#    tt = data[datestr]
#    mag = data[magstr]
#    magerr = data[magerrstr]
#    kwargs = dict(generalized=generalized)
#    omega, power = search_frequencies(tt, mag, magerr, n_eval=n_eval,
#                                      n_retry=n_retry,
#                                      LS_kwargs=kwargs)
#    
#    period = 2*np.pi / omega[np.argmax(power)]
#    
#    return period
