from radmc3dPy import *
import numpy as np
import glob
import matplotlib.pyplot as plb
import os
import fargo3d as fp

au = 1.496e13 #cm
os.chdir('/data/cas213/hltau120_0.066_0.25_results/')
im = image.readImage()
cim = im.imConv(dpc=140., fwhm=[0.05, 0.05], pa=0.)

r_new = np.linspace(0.,100,400)*au
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

np.save('/home/cas213/project_lent/hltau120_066_Ierg_conv_av.npy',I_new_conv)
np.save('/home/cas213/project_lent/hltau120_066_Ierg_av.npy',I_new)
np.save('/home/cas213/project_lent/hltau120_066_r_I_av.npy',r_new_c)




os.chdir('/data/cas213/hltau12_0.05_0.25_results/')
im = image.readImage()
cim = im.imConv(dpc=140., fwhm=[0.05, 0.05], pa=0.)

r_new = np.linspace(0.,100,400)*au
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

np.save('/home/cas213/project_lent/hltau20_05_Ierg_conv_av.npy',I_new_conv)
np.save('/home/cas213/project_lent/hltau20_05_Ierg_av.npy',I_new)
np.save('/home/cas213/project_lent/hltau20_05_r_I_av.npy',r_new_c)



