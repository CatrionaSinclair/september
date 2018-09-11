from radmc3dPy import *
import numpy as np
import glob
import matplotlib.pyplot as plb
import os
import fargo3d as fp

"""
Disk Temperature:
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:07:33 2017

@author: catrionasinclair
"""
from radmc3dPy import *
import numpy as np
import glob
import matplotlib.pyplot as plb
import os
import fargo3d as fp
import scipy.interpolate as itp

au = 14959787070000.
#################################################################################################
stellarmass = '0.3*ms'
stellarradius = '2.32*rs'
diskradius = '48.*au'
diskmass = '0.0021*ms'
stellartemp = '3360'
innerradius = '0.48*au'
r_image = 48.0
Mstar = 0.3
analyze.writeDefaultParfile('ppdisk_temp')   
setup.problemSetupDust('ppdisk_temp', nphot = 2e7, gap_drfact='[0]', rin = innerradius, nz='0',xbound=[innerradius,diskradius], modified_random_walk=1, nx = [225],plh=0.25,mstar=stellarmass, rstar=stellarradius, tstar=stellartemp, mdisk=diskmass, rdisk=diskradius)
par = analyze.readParams()
par.printPar()
os.system('/home/cas213/project_lent/radmc-3d/version_0.41/src/radmc3d mctherm')
data = analyze.readData(dtemp=True)
T0 = data.dusttemp
temp = np.zeros([len(T0[:,0,0]),len(T0[:,0,0])],dtype=float)
for i in range(len(T0[:,0,0])):
    for j in range(len(T0[0,:,0])):
	temp[i,j] = T0[i,j,:,0].mean()
np.save('/home/cas213/project_lent/00_3Ms_dusttemp_hr.npy',temp.T)
np.save('/home/cas213/project_lent/00_3Ms_xgrid_hr.npy',data.grid.x)
np.save('/home/cas213/project_lent/00_3Ms_ygrid_hr.npy',data.grid.y)
plb.figure()
c = plb.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 30)
plb.xlabel('r [AU]')
plb.ylabel(r'$\pi/2-\theta$')
plb.xscale('log')
cb = plb.colorbar(c)
cb.set_label('T [K]', rotation=270.)
c = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 10,  colors='k', linestyles='solid')
plb.clabel(c, inline=1, fontsize=10)
plb.savefig('/home/cas213/project_lent/00_3Ms_temp.png')
image.makeImage(npix=2000., wav=850, incl=0., phi=0., sizeau=150.)
im = image.readImage()
cim = im.imConv(dpc=140., fwhm=[0.05, 0.05], pa=0.)
r_new = np.linspace(0.,r_image,400)*au
diff = (r_new[-1]-r_new[0])/len(r_new)
r_new_c = np.linspace(diff,(r_new[-1]-diff),(len(r_new)-1))
I_new = np.zeros([len(r_new_c)],dtype=float)
I_new_conv = np.zeros([len(r_new_c)],dtype=float)
r = np.zeros([len(im.x),len(im.y)],dtype=float)
print(r_new[0],r_new[-1],r_new_c[0],r_new_c[-1])
for ix1 in range(len(im.x)):
    for iy in range(len(im.y)):
        r[ix1,iy] = np.sqrt(im.x[ix1]**2 + im.y[iy]**2)
for ir in range(len(r_new_c)):
    Idum=np.empty(0,dtype=float)
    Icdum=np.empty(0,dtype=float)
    ii = ((r>=r_new[ir])&(r<r_new[ir+1]))
    if ii.__contains__(True):
        Idum = np.append(Idum, im.image[ii,0])
        Icdum = np.append(Icdum, cim.image[ii,0])
    if len(Idum) != 0 :
	I_new[ir] = Idum.sum()/len(Idum)
        I_new_conv[ir] = Icdum.sum()/len(Icdum)
np.save('/home/cas213/project_lent/Ms03_Ierg_conv_av.npy',I_new_conv)
np.save('/home/cas213/project_lent/Ms03_Ierg_av.npy',I_new)
np.save('/home/cas213/project_lent/Ms03_r_I_av.npy',r_new_c)



######################################################################
stellarmass = '0.7*ms'
stellarradius = '2.54*rs'
diskradius = '80.5*au'
diskmass = '0.00625*ms'
stellartemp = '4024'
innerradius = '0.805*au'
r_image = 80.5
Mstar = 0.7
analyze.writeDefaultParfile('ppdisk_temp')   
setup.problemSetupDust('ppdisk_temp', nphot = 2e7, gap_drfact='[0]', rin = innerradius, nz='0',xbound=[innerradius,diskradius], modified_random_walk=1, nx = [225],plh=0.25,mstar=stellarmass, rstar=stellarradius, tstar=stellartemp, mdisk=diskmass, rdisk=diskradius)
par = analyze.readParams()
par.printPar()
os.system('/home/cas213/project_lent/radmc-3d/version_0.41/src/radmc3d mctherm')
data = analyze.readData(dtemp=True)
T0 = data.dusttemp
temp = np.zeros([len(T0[:,0,0]),len(T0[:,0,0])],dtype=float)
for i in range(len(T0[:,0,0])):
    for j in range(len(T0[0,:,0])):
	temp[i,j] = T0[i,j,:,0].mean()
np.save('/home/cas213/project_lent/00_7Ms_dusttemp_hr.npy',temp.T)
np.save('/home/cas213/project_lent/00_7Ms_xgrid_hr.npy',data.grid.x)
np.save('/home/cas213/project_lent/00_7Ms_ygrid_hr.npy',data.grid.y)
plb.figure()
c = plb.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 30)
plb.xlabel('r [AU]')
plb.ylabel(r'$\pi/2-\theta$')
plb.xscale('log')
cb = plb.colorbar(c)
cb.set_label('T [K]', rotation=270.)
c = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 10,  colors='k', linestyles='solid')
plb.clabel(c, inline=1, fontsize=10)
plb.savefig('/home/cas213/project_lent/00_7Ms_temp.png')
image.makeImage(npix=2000., wav=850, incl=0., phi=0., sizeau=200.)
im = image.readImage()
cim = im.imConv(dpc=140., fwhm=[0.05, 0.05], pa=0.)
r_new = np.linspace(0.,r_image,400)*au
diff = (r_new[-1]-r_new[0])/len(r_new)
r_new_c = np.linspace(diff,(r_new[-1]-diff),(len(r_new)-1))
I_new = np.zeros([len(r_new_c)],dtype=float)
I_new_conv = np.zeros([len(r_new_c)],dtype=float)
r = np.zeros([len(im.x),len(im.y)],dtype=float)
print(r_new[0],r_new[-1],r_new_c[0],r_new_c[-1])
for ix1 in range(len(im.x)):
    for iy in range(len(im.y)):
        r[ix1,iy] = np.sqrt(im.x[ix1]**2 + im.y[iy]**2)
for ir in range(len(r_new_c)):
    Idum=np.empty(0,dtype=float)
    Icdum=np.empty(0,dtype=float)
    ii = ((r>=r_new[ir])&(r<r_new[ir+1]))
    if ii.__contains__(True):
        Idum = np.append(Idum, im.image[ii,0])
        Icdum = np.append(Icdum, cim.image[ii,0])
    if len(Idum) != 0 :
	I_new[ir] = Idum.sum()/len(Idum)
        I_new_conv[ir] = Icdum.sum()/len(Icdum)
np.save('/home/cas213/project_lent/Ms07_Ierg_conv_av.npy',I_new_conv)
np.save('/home/cas213/project_lent/Ms07_Ierg_av.npy',I_new)
np.save('/home/cas213/project_lent/Ms07_r_I_av.npy',r_new_c)



######################################################################
stellarmass = '1.0*ms'
stellarradius = '2.62*rs'
diskradius = '100.*au'
diskmass = '0.01*ms'
stellartemp = '4278'
innerradius = '1.*au'
r_image = 100.
Mstar = 1.0
analyze.writeDefaultParfile('ppdisk_temp')   
setup.problemSetupDust('ppdisk_temp', nphot = 2e7, gap_drfact='[0]', rin = innerradius, nz='0',xbound=[innerradius,diskradius], modified_random_walk=1, nx = [250],plh=0.25,mstar=stellarmass, rstar=stellarradius, tstar=stellartemp, mdisk=diskmass, rdisk=diskradius)
par = analyze.readParams()
par.printPar()
os.system('/home/cas213/project_lent/radmc-3d/version_0.41/src/radmc3d mctherm')
data = analyze.readData(dtemp=True)
T0 = data.dusttemp
temp = np.zeros([len(T0[:,0,0]),len(T0[:,0,0])],dtype=float)
for i in range(len(T0[:,0,0])):
    for j in range(len(T0[0,:,0])):
	temp[i,j] = T0[i,j,:,0].mean()
np.save('/home/cas213/project_lent/01_0Ms_dusttemp.npy',temp.T)
np.save('/home/cas213/project_lent/01_0Ms_xgrid.npy',data.grid.x)
np.save('/home/cas213/project_lent/01_0Ms_ygrid.npy',data.grid.y)
plb.figure()
c = plb.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 30)
plb.xlabel('r [AU]')
plb.ylabel(r'$\pi/2-\theta$')
plb.xscale('log')
cb = plb.colorbar(c)
cb.set_label('T [K]', rotation=270.)
c = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 10,  colors='k', linestyles='solid')
plb.clabel(c, inline=1, fontsize=10)
plb.savefig('/home/cas213/project_lent/01_0Ms_temp.png')

image.makeImage(npix=2000., wav=850, incl=0., phi=0., sizeau=300.)
im = image.readImage()
cim = im.imConv(dpc=140., fwhm=[0.05, 0.05], pa=0.)
r_new = np.linspace(0.,r_image,400)*au
diff = (r_new[-1]-r_new[0])/len(r_new)
r_new_c = np.linspace(diff,(r_new[-1]-diff),(len(r_new)-1))
I_new = np.zeros([len(r_new_c)],dtype=float)
I_new_conv = np.zeros([len(r_new_c)],dtype=float)
r = np.zeros([len(im.x),len(im.y)],dtype=float)
print(r_new[0],r_new[-1],r_new_c[0],r_new_c[-1])
for ix1 in range(len(im.x)):
    for iy in range(len(im.y)):
        r[ix1,iy] = np.sqrt(im.x[ix1]**2 + im.y[iy]**2)
for ir in range(len(r_new_c)):
    Idum=np.empty(0,dtype=float)
    Icdum=np.empty(0,dtype=float)
    ii = ((r>=r_new[ir])&(r<r_new[ir+1]))
    if ii.__contains__(True):
        Idum = np.append(Idum, im.image[ii,0])
        Icdum = np.append(Icdum, cim.image[ii,0])
    if len(Idum) != 0 :
	I_new[ir] = Idum.sum()/len(Idum)
        I_new_conv[ir] = Icdum.sum()/len(Icdum)
np.save('/home/cas213/project_lent/Ms10_Ierg_conv_av.npy',I_new_conv)
np.save('/home/cas213/project_lent/Ms10_Ierg_av.npy',I_new)
np.save('/home/cas213/project_lent/Ms10_r_I_av.npy',r_new_c)

######################################################################



"""
data = analyze.readData(dtemp=True)
data2 = analyze.readData(ddens=True)
data2.getSigmaDust()

tau = data2.getTau(wav = 850, axis = 'xyz')

np.save('/home/cas213/project_lent/Ms03_tau.npy',tau[:,0])

Opac= [0.3966,0.3966,0.3967,0.3971,0.4013,0.4637,1.6962,82.7126,15.6877,5.2216]
data3 = analyze.readData(ddens=True)
tau = np.zeros([data2.grid.nx, data2.grid.nz],dtype = np.float64)
for i in range(10):
    data3.getSigmaDust(idust = i)
    sigma = data3.sigmadust
    tau = tau + sigma*Opac[i]
np.save('/home/cas213/project_lent/Ms03_tau.npy',tau[:,0])
#################################################################################
image.makeImage(npix=2000., wav=850, incl=0., phi=0., sizeau=300.)
im = image.readImage()

x = im.x
dum_arr_min = np.where(x/natconst.au > 1)
dum_ind_min = dum_arr_min[0]
minind = dum_ind_min[0]
dum_arr_max = np.where(x/natconst.au > 100)
dum_ind_max = dum_arr_max[0] - 1
maxind = dum_ind_max[0]
image.plotImage(im, au=True, log=True, maxlog=4, saturate=1e-1, cmap=plb.cm.gist_heat)

plb.figure()
plb.plot(im.x[minind:maxind]/natconst.au,im.image[minind:maxind,1000,0])
plb.xlabel('r [AU]')
plb.ylabel('Intensity')
plb.yscale('log')
plb.yscale('log')
plb.savefig('hltau20_04_intensity_Md01.png')


np.save('hltau20_04_xgrid_Md01.npy', im.x[minind:maxind])
np.save('hltau20_04_Intensity_Md01.npy', im.image[minind:maxind,1000,0])
np.save('hltau20_04_Intensity_Jy_Md01.npy', im.imageJyppix[minind:maxind,1000,0])


image.plotImage(im, au=True, log=True, maxlog=3, saturate=1e-1, cmap=plb.cm.gist_heat)
image.plotImage(im, au=True, log=True, maxlog=5, saturate=1e-1, cmap=plb.cm.gist_heat)
data = analyze.readData(dtemp=True)

plb.figure()
c = plb.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 30)plb.xlabel('r [AU]')plb.ylabel(r'$\pi/2-\theta$')plb.xscale('log')cb = plb.colorbar(c)cb.set_label('T [K]', rotation=270.)c = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 10,  colors='k', linestyles='solid')plb.clabel(c, inline=1, fontsize=10)
plb.savefig('hltau20_04_temp_Md01.png')


Opac= [0.3966,0.3966,0.3967,0.3971,0.4013,0.4637,1.6962,82.7126,15.6877,5.2216]
data3 = analyze.readData(ddens=True)
data3.getTau(wav = 850, axis = 'xyz')
tau = np.zeros([data3.grid.nx, data3.grid.nz],dtype = np.float64)
for i in range(10):
    data3.getSigmaDust(idust = i)
    sigma = data3.sigmadust
    tau = tau + sigma*Opac[i]
plb.figure()
plb.plot(data3.grid.x/natconst.au, tau[:,0])
np.save('hltau20_04_tau_Md01.npy',tau[:,0])
plb.xlabel('r [AU]')plb.ylabel('Optical Depth (vertical)')plb.xscale('log')
plb.savefig('hltau20_04_tau_Md01.png')

analyze.writeDefaultParfile('ppdisk')
setup.problemSetupDust('ppdisk', mdisk='1e-5*ms', gap_rin='[10.0*au]', gap_rout='[40.*au]', gap_drfact='[1e-5]', nz='0')
analyze.writeDefaultParfile('ppdisk_hydrofargo')   
par = analyze.readParams()
par.printPar() 
setup.problemSetupDust('ppdisk_hydrofargo', mstar = '[1.0*ms]', rstar='[2.62*rs]', tstar='[4278.13]', mdisk='0.05*ms',nphot=8e7, rin='1.0*au', rdisk='100.0*au',nx='[100,150]', ny= '[5,40,40,5]', xbound ='[1.0*au,10.0*au,100.0*au]', nz=250, ngs = '10', hrpivot = '19.8*au', hrdisk = '0.05', modified_random_walk=1, filepath ="'/data/rosotti/hltau/hltau20_0.05_0.25/'", Rpl = '19.8*au')
"""
