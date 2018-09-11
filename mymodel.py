"""
# 021117
# Model setup for disk
"""
#Import modules
from radmc3dPy import*
import matplotlib.pyplot as plt
#read model structure, use read data function in analyze module
#default = everything false, set true to read physical variable an spatial grid.
data = analyze.readData(ddens=True)
analyze.plotSlice2D(data, var='ddens',plane='xy',log=True,linunit='au')
plt.xscale('log')
plt.show()

#compare sub mm plots to surface density plots
#meeting next tuesday at 3pm

image.makeImage(npix=300,wav=22,incl=0,sizeau=300)
im22000=image.readImage()
image.plotImage(im22000, au=True, log=True, maxlog=10,saturate=1e5,cmap=plt.cm.gist_heat)
im22000.writeFits('image22000nm.fits')

image.makeImage(npix=300,wav=2.2,incl=0,sizeau=300)
im2200=image.readImage()
image.plotImage(im2200, au=True, log=True, maxlog=10,saturate=1e5,cmap=plt.cm.gist_heat)
im2200.writeFits('image2200nm.fits')

image.makeImage(npix=300,wav=0.22,incl=0,sizeau=300)
im220=image.readImage()
image.plotImage(im220, au=True, log=True, maxlog=10,saturate=1e5,cmap=plt.cm.gist_heat)
im220.writeFits('image220nm.fits')
