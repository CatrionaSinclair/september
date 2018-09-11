from radmc3dPy import *
import numpy as np
import glob
import matplotlib.pyplot as plb
import os
import fargo3d as fp
hr = '0.1'
r0 = '29.5*au'
r_image = 48.
Mstar = 0.3
au = 14959787070000.

stellarmass = '0.3*ms'
stellarradius = '2.32*rs'
diskradius = '48.*au'
diskmass = '0.00212*ms'
stellartemp = '3454'
innerradius = '0.48*au'
xbound = '1.5*au'
analyze.writeDefaultParfile('ppdisk')
setup.problemSetupDust('ppdisk', mdisk=diskmass, nphot=1e7, rin=innerradius, rdisk=diskradius,nx='[25,100]', ny= '[5,25,25,5]', xbound =[innerradius,xbound,diskradius], nz=100, ngs = '10',hrpivot = r0, hrdisk = hr,  Rpl = r0)
analyze.writeDefaultParfile('ppdisk_hydrofargo')   

setup.problemSetupDust('ppdisk_hydrofargo', mstar = stellarmass, rstar=stellarradius, tstar=stellartemp, mdisk=diskmass, nphot=2e7, rin=innerradius, rdisk=diskradius,nx='[75,150]', ny= '[5,40,40,5]', xbound =[innerradius,xbound,diskradius], nz=200, ngs = '10', hrpivot = r0, hrdisk = hr, modified_random_walk=1, filepath ="'/data/rosotti/hltau/hltau60_0.1_0.25/'", Rpl = r0)
par = analyze.readParams()
par.printPar()
data2 = analyze.readData(ddens=True)
data2.getSigmaDust()
np.save('x_Ms03_hltau60_1.npy', data2.grid.x/natconst.au)

os.system('/home/cas213/project_lent/radmc-3d/version_0.41/src/radmc3d mctherm')

image.makeImage(npix=2000., wav=850, incl=0., phi=0., sizeau=300.)

im = image.readImage()
cim = im.imConv(dpc=140., fwhm=[0.05, 0.05], pa=0.)

image.plotImage(im, au=True, log=True, maxlog=4, saturate=1e-1, cmap=plb.cm.gist_heat)
image.plotImage(im, au=True, log=True, maxlog=4, saturate=1e-2, cmap=plb.cm.gist_heat)

image.plotImage(cim, au=True, log=True, maxlog=4, saturate=1e0, cmap=plb.cm.gist_heat)
image.plotImage(cim, au=True, log=True, maxlog=4, saturate=1e-1, cmap=plb.cm.gist_heat)

r_new = np.linspace(0.,r_image,400)*au
diff = (r_new[-1]-r_new[0])/len(r_new)
r_new_c = np.linspace(diff,(r_new[-1]-diff),(len(r_new)-1))
I_new = np.zeros([len(r_new_c)],dtype=float)
I_new_conv = np.zeros([len(r_new_c)],dtype=float)
r = np.zeros([len(im.x),len(im.y)],dtype=float)

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

np.save('/home/cas213/project_lent/Ms03_hltau60_1_Ierg_conv_av.npy',I_new_conv)
np.save('/home/cas213/project_lent/Ms03_hltau60_1_Ierg_av.npy',I_new)
np.save('/home/cas213/project_lent/Ms03_hltau60_1_r_I_av.npy',r_new_c)

data = analyze.readData(dtemp=True)
data2 = analyze.readData(ddens=True)
data2.getSigmaDust()

data2.getTau(wav = 850, axis = 'xyz')

Opac= [0.3966,0.3966,0.3967,0.3971,0.4013,0.4637,1.6962,82.7126,15.6877,5.2216]
data3 = analyze.readData(ddens=True)
tau = np.zeros([data2.grid.nx, data2.grid.nz],dtype = np.float64)
for i in range(10):
    data3.getSigmaDust(idust = i)
    sigma = data3.sigmadust
    tau = tau + sigma*Opac[i]
np.save('Ms03_hltau60_1_tau.npy',tau[:,0])

