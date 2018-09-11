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
import fargoPy as fp

analyze.writeDefaultParfile('ppdisk')    
setup.problemSetupDust('ppdisk_hydro', mdisk='0*ms',nphot=1e6)
par = analyze.readParams()
#par.printPar()
data2 = analyze.readData(ddens=True)
data2.getSigmaDust()

fargodata = fp.read_fargo_data(path='/home/cas213/project/popsint_spiral_hp0p1_1mjup', frame=4, sigma=True, ir = data2.grid.x/(50.0*natconst.au), iphi=data2.grid.z)
sigma1 = fargodata['isigma'] # (phi, r)
fp.plot_data(fargodata, var='sigma', projection ='cartesian',stretch='log')
plb.show() # x = radial, y = theta, z = phi,

#os.system('/home/cas213/project/radmc-3d/version_0.41/src/radmc3d mctherm')

image.makeImage(npix=1000., wav=850, incl=0., phi=0., sizeau=300.)
im = image.readImage()
image.plotImage(im, au=True, log=True, maxlog=7, saturate=1e-5, cmap=plb.cm.gist_heat)

"""
data = analyze.readData(dtemp=True)
c = plb.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 30)
plb.xlabel('r [AU]')
plb.ylabel(r'$\pi/2-\theta$')
plb.xscale('log')
cb = plb.colorbar(c)
cb.set_label('T [K]', rotation=270.)
c = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 10,  colors='k', linestyles='solid')
plb.clabel(c, inline=1, fontsize=10)
plb.show()"""
