import numpy as np
import struct as st
import os, sys 
from matplotlib.pyplot import *
import matplotlib.cm as cm

class fargo3d():
    """
    Class to read data from fargo3D

    Grid notation:
        r       - radial coordinate
        phi     - azimuthal angular coordinate
        theta   - poloidal angular coordinate
        rc      - radial coordinate of cell centers
        phic    - azimuthal coordinate of cell centers
        thetac  - poloidal coordinate of cell centers
        ri      - radial coordinate of cell interfaces/walls
        phii    - azimuthal coordinate of cell interfaces/walls
        thetai  - poloidal coordinate of cell interfaces/walls
    """

    def __init__(self):

        self.nr     = None
        self.nphi   = None
        self.ntheta = None
        self.rc     = None
        self.phic   = None
        self.thetac = None
        self.ri     = None
        self.phii   = None
        self.thetai = None
        self.gridhead = None
        
        self.sigmadust = None
        self.sigmagas  = None
        self.vrad      = None
        self.vphi      = None
        self.vtheta    = None

        self.stokesNr  = None
        self.gsize     = None
        self.ndust     = None

        self.vx        = None
        self.vy        = None
        self.vz        = None
        self.vrot      = None

    def readDims(self, fileName='dimensions.dat', path='./'):
        """
        Reads the dimensions of the simulations frmo dimensions.dat
        """
        
        try :
            dimFile = open(path+fileName,'r')
        except:
            print 'File not found - '+fileName
            sys.exit(0)

        hdr     = dimFile.readline()[1:].split()
        line    = dimFile.readline()
        par     = [float(s) for s in line.split()]
        dimFile.close()

        self.gridhead = {}
        for i in range(len(hdr)):
            self.gridhead[hdr[i]] = par[i]

        self.nr = self.gridhead['NY']
        self.nphi = self.gridhead['NX']
        self.ntheta = self.gridhead['NZ']
        
        
    def readGrid(self, path='./'):
        """
        Reads the spatial grid from the domain_xyz.dat files
        """

        #
        # Get the radial coordinates
        #
        self.ri     = np.genfromtxt(path+'/domain_y.dat')[int(self.gridhead['NGHY']):-int(self.gridhead['NGHY'])]
        self.rc     = 0.5 * (self.ri[1:] + self.ri[:-1])
        self.phii   = np.genfromtxt(path+'/domain_x.dat')
        self.phic   = 0.5 * (self.phii[1:] + self.phii[:-1])
        self.thetai = np.genfromtxt(path+'/domain_z.dat')[int(self.gridhead['NGHZ']):-int(self.gridhead['NGHZ'])]
        self.thetac = 0.5 * (self.thetai[1:] + self.thetai[:-1])
         
    def readData(self, frame=0, path='./', ddens=False, gdens=False, gvel=False, idust=-1):
        """
        Reads the data fields of the simulations
        """
        self.readDims(path=path)
        self.readGrid(path=path)
        
        self.Ndust = -1
        if ddens:

            self.stokesNr = 1. / np.genfromtxt(path+'/drag_coefficients.dat')
            if type(self.stokesNr).__name__!='ndarray':
                self.stokesNr = np.array([self.stokesNr])

            self.Ndust = len(self.stokesNr)
           
            self.sigmadust = []
            if self.ntheta<=1:
                self.sigmadust = []
                if idust>=0:
                    self.stokesNr = self.stokesNr[idust]
                    self.Ndust    = 1
                    print 'Reading dust size : 0'
                    self.sigmadust.append(np.fromfile(path+'/dustdens'+str(idust)+'_'+str(frame)+'.dat').reshape(self.nr, self.nphi))

                else:
                    for i in range(self.Ndust):
                        print 'Reading dust size : ', i
                        self.sigmadust.append(np.fromfile(path+'/dustdens'+str(i)+'_'+str(frame)+'.dat').reshape(self.nr, self.nphi))
           
            else:
                print 'This seems to be a 3D model'
                print 'Reading from 3D models are not yet implemented'

        if gdens:

            if self.ntheta<=1:
                self.sigmagas = np.fromfile(path+'/gasdens'+str(frame)+'.dat').reshape(self.nr, self.nphi)

            else:
                print 'This seems to be a 3D model'
                print 'Reading from 3D models are not yet implemented'


        if gvel:

            if self.ntheta<=1:
                print 'WARNING: it is assumed that the frame in fargo is corotating and the planet orbital radius is 1.0!!!'
                dum = raw_input('Is this OK (y/n)?')
                if dum.strip().lower()[0] == 'y':
                    # The frame rotates with Omega_pl = 1./sqrt(r) 
                    # Since r=1, Omega_pl = 1
                    # Now we need to add r*Omega_pl to the azimuthal velocity
                    self.vx = np.fromfile(path+'/gasvx'+str(frame)+'.dat').reshape(self.nr, self.nphi) 
                    self.vy = np.fromfile(path+'/gasvy'+str(frame)+'.dat').reshape(self.nr, self.nphi) 
                    self.vrot = np.zeros([self.nr, self.nphi], dtype=float)
                    for i in range(int(self.nr)):
                        self.vrot[i,:] = self.rc[i]
                else:
                    return
                

    def fargoData(self, frame=0, path='./', ddens=False, gdens=False, gvel=False, idust=-1):
        """
        Fargo3D analogue of the fargo_data function for the 2D legacy code
        
        Parameters
        ----------
        path    : str
                  Path to the fargo3d data files

        frame   : int
                  Frame number to be read

        ddens   : bool
                  Should the dust density be read?

        gdens   : bool
                  Should the gas density be read?

        gvel    : bool
                  Should the gas velocity be read?

        idust   : int
                  Dust species index to be read. If -1 all dust species will be read at once
        """

        self.readData(frame=frame,path=path, ddens=ddens,gdens=gdens,gvel=gvel,idust=idust)

        res = {}
        res['rc']        = self.rc
        res['ri']        = self.ri
        res['phic']      = self.phic
        res['phii']      = self.phii
        res['thetac']    = self.thetac
        res['thetai']    = self.thetai
        res['nr']        = self.nr
        res['nphi']      = self.nphi
        res['ntheta']    = self.ntheta
        res['sigmadust'] = self.sigmadust
        res['sigmagas']  = self.sigmagas
        res['stokesNr']  = self.stokesNr
        res['Ndust']     = self.Ndust
        res['vphi']      = self.vx
        res['vrad']      = self.vy
        res['vrot']      = self.vrot
        
        return res

def read_fargo_data(path='./', frame=0, ddens=False, gdens=False, gvel=False, idust=-1):
    """
    Reads fargo3d data

    Parameters
    ----------
    path    : str
              Path to the fargo3d data files

    frame   : int
              Frame number to be read

    ddens   : bool
              Should the dust density be read?

    gdens   : bool
              Should the gas density be read?

    gvel    : bool
              Should the gas velocity be read?

    idust   : int
              Dust species index to be read. If -1 all dust species will be read at once
    """

    f = fargo3d()
    res = f.fargoData()
    return res
"""
## =================================================================================================
##   Test
## =================================================================================================
d = fargo3d()
import os
os.chdir('hltau8_0.05_0.0001')
d.readDims()
d.readGrid()
d.readData(frame=600, ddens=True, gdens=True)
print(d.sigmadust[0].shape)
imshow(np.log10(d.sigmadust[0]), extent=(d.rc[0], d.rc[-1], d.phic[0], d.phic[-1]), aspect='auto',\
        vmin=-4., vmax=-2.)
colorbar()
show()

fig = figure()
rr,pp = np.meshgrid(d.rc, d.phic)
xx = rr * np.cos(pp)
yy = rr * np.sin(pp)
pcolormesh(xx,yy, d.sigmadust[1].T, vmin=0, vmax=0.0005)
colorbar()
show()

fig = figure()
rr,pp = np.meshgrid(d.rc, d.phic)
xx = rr * np.cos(pp)
yy = rr * np.sin(pp)
pcolormesh(xx,yy, d.sigmagas.T, vmin=0, vmax=0.0005)
colorbar()
show()

dum = raw_input()
"""                    
