# -*- coding: utf-8 -*-
"""
First attempt at (re)creating some of the basic "Tom tools" I have made in matlab.

Created on Fri Jun 12 17:03:58 2020

@author: jtomf
jfarrar@whoi.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from scipy import fft
from scipy import stats
from scipy import signal
import matplotlib
import time


def running_average(f, N):
    """ Calculate N-point moving mean
          of the array f.

         return
            - averaged f [same size as f]
        Cesar Rocha made this.
    """

    cumsum = np.cumsum(np.insert(f,0,0))
    return (cumsum[N:] - cumsum[:-N]) / N

def run_avg1d(f, N):
    """
    Calculate N-point running average of the 2d array f.      
 
    Parameters
    ----------
    f : numeric
        So far I have only used a 1d Numpy.array.
    N : numeric
        Number of points for running average
    dim : numeric
        dimension on which to operate (1 or 2, default 1)

    Returns
    -------
    result : numeric
        running average of f [same size as f].

    Example
    -------
    >>> x = np.linspace(0, 100, 1000)
    >>> y = np.linspace(0, 80, 800)
    >>> xx, yy = np.meshgrid(x, y)
    >>> z=np.cos(2*np.pi/10*xx+2*np.pi/19*yy)+np.random.normal(0,1,np.shape(xx))
    >>> z[400, 400] = np.nan
    >>> N = 51
    >>> fz = run_avg2d(z, N, 1)
    >>> fz2 = run_avg2d(fz, N, 2)

    """
    win = np.ones((N,))
    sumwin = sum(win)
    # Initialize fz
    fz = np.empty(np.shape(f))
    fz = np.convolve(f, win, mode='same') 

    return fz / sumwin

def run_avg2d(f, N, dim):
    """
    Calculate N-point running average of the 2d array f.      
 
    Parameters
    ----------
    f : numeric
        So far I have only used a 2d Numpy.array.
    N : numeric
        Number of points for running average
    dim : numeric
        dimension on which to operate (1 or 2, default 1)

    Returns
    -------
    result : numeric
        running average of f [same size as f].

    Example
    -------
    >>> x = np.linspace(0, 100, 1000)
    >>> y = np.linspace(0, 80, 800)
    >>> xx, yy = np.meshgrid(x, y)
    >>> z=np.cos(2*np.pi/10*xx+2*np.pi/19*yy)+np.random.normal(0,1,np.shape(xx))
    >>> z[400, 400] = np.nan
    >>> N = 51
    >>> fz = run_avg2d(z, N, 1)
    >>> fz2 = run_avg2d(fz, N, 2)

    """
    win = np.ones((N,))
    sumwin = sum(win)
    # Initialize fz
    fz = np.empty(np.shape(f))
    if dim == 1:
        for n in range(0, len(f[0, :])-1):
            fz[:, n] = np.convolve(f[:, n], win, mode='same') 

    elif dim == 2:
        for n in range(0, len(f[:, 0])-1):
           fz[n, :] = np.convolve(f[n, :], np.transpose(win), mode='same') 

    return fz / sumwin

def centeredFFT(x, dt):
    """
    Computes FFT, with zero frequency in the center, and returns 
    dimensional frequency vector.
    X, freq = centeredFFT(x, dt)
    
    Parameters
    ----------
    x : numpy.ndarray of shape (N,) or (N,1)
        1D array to be transformed by FFT
    dt : numeric
        Time increment (used to compute dimensional freuency array)

    Returns (tuple)
    -------
    X: FFT of input x, with zero frequency in the center
    freq: Dimensional frequency vector corresponding to X

    #function [X,freq]=centeredFFT(x,dt)
    #
    # Adapted from a matlab function written by Quan Quach of blinkdagger.com 
    # Tom Farrar, 2016, 2020 jfarrar@whoi.edu
    # converted from matlab 2020
    # This code was written for MIT 12.805 class
    """
    N = len(x)
    x = x.reshape(N,)
    
    #Generate frequency index
    if N % 2 == 0:
        m= np.arange(-N/2,N/2,1) # N even; this includes start (-N/2) and does not include stop (+N/2)
        #m = mslice[-N / 2:N / 2 - 1]    # N even (Matlab syntax)
    else:
        m= np.arange(-(N-1)/2,(N-1)/2+1,1) # N odd
        #m = mslice[-(N - 1) / 2:(N - 1) / 2]    # N odd (Matlab syntax)
    
    freq = m / (N * dt) #the dimensional frequency scale
    X = fft.fft(x)
    X = fft.fftshift(X) #swaps the halves of the FFT vector so that the zero frequency is in the center\\
    #If you are going to compute an IFFT, first use X=ifftshift(X) to undo the shift}
    return (X, freq) # Return tuple; could instead do this as dictionary or list

def band_avg(yy, num):
    '''
    Compute block averages for band averaging.

    Parameters
    ----------
    yy : np.array
        1D array to be averaged.
    num : numeric
        number of adjacent data points to average.

    Returns
    -------
    Bin-averaged version of input data, subsampled by the factor num

    # Tom Farrar, 2016, 2020 jfarrar@whoi.edu
    # This code was written for MIT 12.805 class

    '''
    #MATLAB code:
    #yyi=0;
    #for n=1:num
    # yyi=yy(n:num:[end-(num-n)])+yyi;
    #end
    
    yyi = 0
    for n in np.arange(0, num): # 1:num
        yyi=yy[n:-(num-n):num]+yyi;
        
    yy_avg=yyi/num
    
    return yy_avg


# From a Stackoverflow post by Rich Signell
# https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac
# matlab2datetime_vec = np.vectorize(tt.matlab2datetime)
# time = matlab2datetime_vec(mat['mday'])


def matlab_whos(mat):
    '''
    Parameters
    ----------
    mat : dictionary
        Dictionary generated from MATLAB data file using
        mat = scipy.io.loadmat(pth + file)

    Returns
    -------
    None.  (Displays variable names and shapes)

    '''
    for n, nn in enumerate(dict.keys(mat)): print(nn, end=" "), print(np.shape(mat[nn]))

def confid(alpha,nu):
    """
    Computes the upper and lower 100(1-alpha)% confidence limits for 
    a chi-square variate (e.g., alpha=0.05 gives a 95% confidence interval).
    Check values (e.g., Jenkins and Watts, 1968, p. 81) are $\nu/\chi^2_{19;0.025}=0.58$
    and $\nu/\chi^2_{19;0.975}=2.11$ (I get 0.5783 and 2.1333 in MATLAB).
    
   
    Parameters
    ----------
    alpha : numeric
        Number of degrees of freedom
    nu : numeric
        Number of degrees of freedom

    Returns (tuple)
    -------
    lower: lower bound of confidence interval
    upper: upper bound of confidence interval

    # Tom Farrar, 2020, jfarrar@whoi.edu
    # converted from matlab 2020
    # This code was written for MIT 12.805 class
    """
    
    # requires:
    # from scipy import stats
    
    upperv=stats.chi2.isf(1-alpha/2,nu)
    lowerv=stats.chi2.isf(alpha/2,nu)
    lower=nu / lowerv
    upper=nu / upperv
    
    return (lower, upper) # Return tuple; could instead do this as dictionary or list
    
def confidence_interval(alpha,nu,cstr,yspot=None,xspot=None,width=None,ax=None):
    """
    Plot (1-alpha)*100% spectral confidence interval on a log-log scale
    
    Parameters
    ----------
    alpha: numeric, between 0 and 1
        100*alpha is the percentage point of the chi-square distribution
        For example, use alpha=0.05 for a 95% confidence interval
    nu: numeric
        number of degrees of freedom
    cstr:
        color for confidence interval 
        For example, cstr = 'r' or cstr = h[-1].get_color()
    xspot: (1,) numpy.ndarray
        horizontal coordinate for confidence interval (e.g., xspot=freq(3);)
    yspot: (1,) numpy.ndarray
        vertical coordinate for confidence interval
    width: numeric
        width (in points) for top and bottom horizontal bars


    Returns
    -------
 
    # Tom Farrar, 2020, jfarrar@whoi.edu
    # converted from matlab 2020
    # This code was written for MIT 12.805 class
 """
    if ax is None:
      ax = plt.gca()
    if width is None:
      plt.sca(ax)
      fig = plt.gcf()
      # Get size of figure in pixels (would be preferable to use axis size instead)
      size = fig.get_size_inches()*72 # size in points (matplotlib uses 72 points per inch, https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size)
      width = np.round(0.0075*size[0]) # make cap width fixed % of figure width
      # Get width of axis in pixels
      # ax_width = np.diff(ax.get_xlim())*fig.dpi
      # width = np.round(0.015*ax_width) # make cap width 1.5% of figure width
    if yspot is None:
      plt.sca(ax)
      yax = ax.get_ylim() # Get xlim of current axis
      yax_midpoint = 0.75*np.diff(np.log10(yax)) # find point 75% from left in log space
      yspot = 10**(yax_midpoint + np.log10(yax[0])) # set default xspot to 75% from left
    if xspot is None:
      plt.sca(ax)
      xax = ax.get_xlim() # Get xlim of current axis
      xax_midpoint = 0.75*np.diff(np.log10(xax)) # find point 75% from left in log space
      xspot = 10**(xax_midpoint + np.log10(xax[0])) # set default xspot to 75% from left

    lower, upper = confid(alpha, nu)

    # Definition of lowz and upz differs from matlab version because 
    # plt.errorbar plots log10(yspot-lowz) and log10(yspot+upz), whereas, in 
    # matlab version I was plotting log10(yspot*lower) and log10(yspot*upper)
    lowz = yspot*(1-lower)
    upz = yspot*(upper-1)
    err = [lowz, upz]

    # plot confidence interval
    plt.errorbar(xspot, yspot, yerr=err, fmt='', capsize=width, ecolor=cstr)
    plt.text(xspot,yspot,'  ' + str(100*(1-alpha)) + '%',horizontalalignment='left');
    # plt.show()
    return(ax)

def spectrum_band_avg(yy,dt,M,winstr=None,plotflag=None,ebarflag=None):
    """
    Make one-sided, band-averaged spectral estimate for a real, 1-D data record
    
    Parameters
    ----------
    yy: numpy.ndarray
        vector to be transformed (must be real and uniformly spaced)
    dt: numeric
        time increment
    M: numeric
        number of bands to average
    winstr: None or string
        Taper window, must be None or one of 'hann', 'blackman', 'parzen', 
  			'triang', 'tukey' (tukey is implemented over first/last 10% of record).
        None uses a rectangular window (i.e., no taper).
    plotflag: None, True, or False
        Plots spectrum if set to True.  Does not plot if set to False or None.

    Returns
    -------
    Spectrum: numpy.ndarray
        A band-averaged estimate of the spectrum.
    freq: numpy.ndarray
        Frequency vector corresponding to the spectrum.
    EDOF: float
        An estimate of the equivalent number of degrees of freedom, after 
        accounting for the use of the taper window (valid for large M)
 
    # Tom Farrar, 2020, jfarrar@whoi.edu
    # converted from matlab 2020
    # This code was written for MIT 12.805 class
 """

    if plotflag is None:
      plotflag = False
    if ebarflag is None:
        ebarflag = True

    N = len(yy)
    yy = yy.reshape(N,)
    T = N * dt
    # Compute the mean of the time series and subtract it
    yy = yy - np.mean(yy)

    # Optional use of taper window:
    if winstr is None:
      winstr = 'boxcar'
      win = eval('signal.' + winstr + '(' + str(N) + ',sym=False)')
      G=1.0
    elif winstr == 'hann':
      win = eval('signal.' + winstr + '(' + str(N) + ',sym=False)')
      G=1.9445
    elif winstr == 'blackmann':
      win = eval('signal.' + winstr + '(' + str(N) + ',sym=False)')
      G=2.3481
    elif winstr == 'parzen':
      win = eval('signal.' + winstr + '(' + str(N) + ',sym=False)')
      G=2.6312
    elif winstr == 'triang':
      win = eval('signal.' + winstr + '(' + str(N) + ',sym=False)')
      G=1.8000
    # elif winstr == 'gauss': # (L – 1)/(2α)--> see matlab doc for gausswin
    #  win = eval('signal.' + winstr + '(' + str(N) + ',sym=False)')
    #  G = 2.7926 # for matlab gaussian window with alpha=2.5
    elif winstr == 'tukey':
      win = eval('signal.' + winstr + '(' + str(N) + ',alpha=0.2,sym=False)')
      G = 1.1561 # for tapered-cosine window with taper applied over first and last 10% (alpha=.2 in scipy.signal)

    win=win/np.sqrt(np.sum(win**2/N))    
    EDOF=2*M/G # equivalent DOF for band average, valid for large M
    yy = win*yy

    # Compute FFT and frequency scale, with lowest frequency in the center
    (Y,freq_i) = centeredFFT(yy,dt)
    # D==card negative frequencies 
    ff = np.where(freq_i>0)
    Y = Y[ff]
    freq_i = freq_i[ff]
    # Apply normalization for 1-sided spectral density
    YY_raw = 2*T/N**2*Y*np.conj(Y) 
    # (note that the order of Y vs np.conj(Y) matters for a cross spectrum)

    #Carry out band averaging
    YY_avg1 = band_avg(YY_raw,M)
    freq = band_avg(freq_i,M)
    
    YY_avg = np.real(YY_avg1)

    nu = float(EDOF)
    alpha = 0.05

    if plotflag is True:
     # with plt.xkcd(.35,100,2):
     #   fig = plt.figure()
        h = plt.loglog(freq, YY_avg)
        plt.title('Band-averaged spectral estimate')
        cstr = h[-1].get_color()
        if ebarflag==True:
            confidence_interval(alpha,nu,cstr)

    return YY_avg, freq, EDOF



'''Clone of matlab tic/toc from Stackoverflow user Benben:
    https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions
'''
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)