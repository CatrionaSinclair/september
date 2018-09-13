#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:41:48 2018

@author: catrionasinclair
"""
import numpy as np
import matplotlib.pylab as plb

au = 14959787070000.
fac = (10.**23)/(360*3600/(2*np.pi))**2


SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 32
plb.rc('font', size=SMALL_SIZE)          # controls default text sizes
plb.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plb.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plb.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plb.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plb.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plb.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def preproc(r0=None, Im0=None, Im_c0=None,Mstar=None,rin=None,rout=None):
    Area = np.pi * (1./1.4)**2 * Mstar**1.22
    I0 = 0.2 * Mstar**0.08 #Jy arcsec^-2
    F0 = I0*Area
    print(F0)
    r = np.linspace(r0[0],r0[-1],5000)
    Im = (np.interp(r,r0,Im0))*fac
    Im_c = (np.interp(r,r0,Im_c0))*fac
    
    x_diff = r[1]-r[0]
    #I_tot = (2*np.pi*r*x_diff*Im*140.**-2).sum()
    I_tot=0.
    for i in range(len(r)):
        dA = 2 * np.pi * (r[i]/140.) * (x_diff/140.)
        I_tot = I_tot + dA * Im[i]
    
    
    Ic_tot = (2*np.pi*r*x_diff*Im_c*140.**-2).sum()
    print(I_tot, Ic_tot)
    Im = Im * F0/I_tot
    Im_c = Im_c * F0/Ic_tot
    while r[0] <= rin :
        r = np.delete(r,0)
        Im = np.delete(Im,0)
        Im_c = np.delete(Im_c,0)
    while r[-1] >= rout :
        r = np.delete(r,-1)
        Im = np.delete(Im,-1)
        Im_c = np.delete(Im_c,-1)
    return(r,Im,Im_c)
    
def GapDepth(r=None, I=None, Ic = None, Mstar=None):
    m = len(r)
    fit1 = np.polyfit(np.log10(r[0:m]),np.log10(I[0:m]),1)
    I_b_func1 = np.poly1d(fit1)
    I_b1 = 10**I_b_func1(np.log10(r))
    
    fit2 = np.polyfit(np.log10(r[0:m]),np.log10(Ic[0:m]),1)
    I_b_func2 = np.poly1d(fit2)
    I_b2 = 10**I_b_func2(np.log10(r))
    plb.figure(figsize=(6,3))
    plb.subplot(121)
    plb.plot(r,I,'r')
    plb.plot(r,I_b1,'r:')
    plb.xscale('log')
    plb.yscale('log')
    plb.subplot(122)
    plb.plot(r,Ic,'b')
    plb.plot(r,I_b2,'b:')
    plb.xscale('log')
    plb.yscale('log')
    
    I_r = I - I_b1
    Ic_r = Ic - I_b2
    #I_r = I - Ib
    #Ic_r = Ic-Ibc
    plb.figure(figsize = (4,4))
    l1, = plb.plot(r,I_r,'r',label='Unconvoled')
   
    ind = np.argmin(I_r)
    plb.plot(r[ind],I_r[ind],'ro')
    if abs(I_r[ind]) >= 0.053:
        lim = I_r[ind]/3
        ii = (I_r <= lim)
        width1 = max(r[ii]) - min(r[ii])
        plb.plot(max(r[ii]),lim,'r*')
        plb.plot(min(r[ii]),lim,'r*')
    else:
        width1 = np.nan
        
    
    ind2 = np.argmin(Ic_r)
    if abs(Ic_r[ind]) >= 0.053:
        lim = Ic_r[ind]/3
        ii = (Ic_r <= lim)
        width2 = max(r[ii]) - min(r[ii])
        plb.plot(max(r[ii]),lim,'b*')
        plb.plot(min(r[ii]),lim,'b*')
    else:
        width2 = np.nan
   
    l2, = plb.plot(r,Ic_r,'b',label='Convolved')
    plb.plot(r[ind2],Ic_r[ind2],'bo')
    plb.xscale('log')
    plb.xlabel('r [AU]')
    plb.ylabel(r'Residual Surface Brightness [Jy arcsec$^{-2}$]',size=10)
    plb.legend()
    plb.plot([r[0],r[-1]],[0,0],'k')
    
    print(' UNCONVOLVED ')
    print('Sb Change   = %.4f' %(abs(I_r[ind])))
    print('Sb Position = %.1f' %(r[ind]))
    print('Gap Width   = %.2f' %(width1))
    print(I[ind],I_b1[ind], I_b2[ind],Ic[ind2])
    print(' ')
    print(' CONVOLVED ')
    print('Sb Change   = %.4f' %(abs(Ic_r[ind2])))
    print('Sb Position = %.1f' %(r[ind2]))
    print('Gap Width   = %.2f' %(width2))

    return()


x0 = np.load('Ms03/Ms03_hltau20_05_r_I_av.npy')/au
I0 = np.load('Ms03/Ms03_hltau20_05_Ierg_av.npy')
Ic0 = np.load('Ms03/Ms03_hltau20_05_Ierg_conv_av.npy')

r, I, Ic = preproc(x0,I0,Ic0,Mstar=.3,rin=1,rout=46.)


plb.figure(figsize=(8,8))
plb.plot(r,I)
plb.plot(r,Ic)
plb.xscale('log')
plb.xlabel('r [AU]')
plb.xlim([1,60])
plb.yscale('log')
plb.ylabel(r'Surface Brightness [Jy arcesec$^{-2}$]')
plb.legend(['Unconvolved Image','Convolved Image'])

x=30
GapDepth(r[x:], I[x:], Ic[x:], Mstar=0.3)
