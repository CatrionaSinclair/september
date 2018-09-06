"""Generic protoplanetary disk model 
The density is given by 
    .. math:
        \\rho = \\frac{\\Sigma(r,\\phi)}{H_p\\sqrt{(2\\pi)}} \\exp{\\left(-\\frac{z^2}{2H_p^2}\\right)}
    * :math:`\Sigma` - surface density
    * :math:`H_{\\rm p}` - Pressure scale height

There are two options for the functional form of surface density as a function of radius. For a simple
power-law the surface density is given by

    * :math:`\Sigma(r) = \\Sigma_0\\left(\\frac{r}{r_{\\rm out}}\\right)^p`

alternatively the surface density can also have an exponential outer tapering:

    * :math:`\Sigma(r) = \\Sigma_0\\left(\\frac{r}{r_{\\rm out}}\\right)^p\\exp{\\left\\{-\\left(\\frac{r}{r_{\\rm out}}\\right)^{2+p}\\right\\}}`


The molecular abundance function takes into account dissociation and freeze-out of the molecules
For photodissociation only the continuum (dust) shielding is taken into account in a way that
whenever the continuum optical depth radially drops below a threshold value the molecular abundance
is dropped to zero. For freeze-out the molecular abundance below a threshold temperature is decreased
by a given fractor. 


"""
from __future__ import absolute_import
from __future__ import print_function
import warnings
import traceback
import fargo3d as fp
import os
import scipy.interpolate as itp
import matplotlib.pylab as plb
try:
    import numpy as np
except ImportError:
    np = None
    print(' Numpy cannot be imported ')
    print(' To use the python module of RADMC-3D you need to install Numpy')
    print(traceback.format_exc())

try:
    import matplotlib.pylab as plb
except ImportError:
    plb = None
    print('Warning')
    print('matplotlib.pylab cannot be imported')
    print('Without matplotlib you can use the python module to set up a model but you will not be able to plot things')
    print('or display images')

from radmc3dPy.natconst import *
from radmc3dPy import analyze


def getModelDesc():
    """Returns the brief description of the model.
    """

    return "Generic protoplanetary disk model"
           

def getDefaultParams():
    """Function to provide default parameter values of the model.

    Returns a list whose elements are also lists with three elements:
    1) parameter name, 2) parameter value, 3) parameter description
    All three elements should be strings. The string of the parameter
    value will be directly written out to the parameter file if requested,
    and the value of the string expression will be evaluated and be put
    to radmc3dData.ppar. The third element contains the description of the
    parameter which will be written in the comment field of the line when
    a parameter file is written. 
    """

    defpar = [
        ['xres_nlev', '3', 'Number of refinement levels'],
        ['xres_nspan', '3', 'Number of the original grid cells to refine'],
        ['xres_nstep', '3', 'Number of grid cells to create in a refinement level'],
        ['nx', '[30,50]', 'Number of grid points in the first dimension'],
        ['xbound', '[1.0*au,1.05*au, 100.0*au]', 'Number of radial grid points'],
        ['ny', '[10,80,80,10]', 'Number of grid points in the first dimension'],
        ['ybound', '[0., pi/3., pi/2., 2.*pi/3., pi]', 'Number of radial grid points'],
        ['nz', '30', 'Number of grid points in the first dimension'],
        ['zbound', '[0., 2.0*pi]', 'Number of radial grid points'],
        ['gasspec_mol_name', "['co']", ''],
        ['gasspec_mol_abun', '[1e-4]', ''],
        ['gasspec_mol_dbase_type', "['leiden']", ''],
        ['gasspec_mol_dissoc_taulim', '[1.0]', 'Continuum optical depth limit below which all molecules dissociate'],
        ['gasspec_mol_freezeout_temp', '[19.0]', 'Freeze-out temperature of the molecules in Kelvin'],
        ['gasspec_mol_freezeout_dfact', '[1e-3]',
         'Factor by which the molecular abundance should be decreased in the frezze-out zone'],
        ['gasspec_vturb', '0.2e5', 'Microturbulent line width'],
        ['rin', '1.0*au', ' Inner radius of the disk'],
        ['rdisk', '100.0*au', ' Outer radius of the disk'],
        ['hrdisk', '0.05', ' Ratio of the pressure scale height over radius at hrpivot'],
        ['hrpivot', "30.0*au", ' Reference radius at which Hp/R is taken'],
        ['plh', '1./4.', ' Flaring index'],
        ['plsig1', '-1.0', ' Power exponent of the surface density distribution as a function of radius'],
        ['sig0', '0.0', ' Surface density at rdisk'],
        ['mdisk', '1e-2*ms', ' Mass of the disk (either sig0 or mdisk should be set to zero or commented out)'],
        ['bgdens', '1e-30', ' Background density (g/cm^3)'],
        ['srim_rout', '0.0', 'Outer boundary of the smoothing in the inner rim in terms of rin'],
        ['srim_plsig', '0.0', 'Power exponent of the density reduction inside of srim_rout*rin'],
        ['prim_rout', '0.0', 'Outer boundary of the puffed-up inner rim in terms of rin'],
        ['hpr_prim_rout', '0.0', 'Pressure scale height at rin'],
        ['gap_rin', '[0e0*au]', ' Inner radius of the gap'],
        ['gap_rout', '[0e0*au]', ' Outer radius of the gap'],
        ['gap_drfact', '[0e0]', ' Density reduction factor in the gap'],
        ['sigma_type', '0',
         ' Surface density type (0 - polynomial, 1 - exponential outer edge (viscous self-similar solution)'],
	['ngs','10', 'Number of Dust Species'],
	['dustkappa_ext','[1,2,3,4,5,6,7,8,9,10]','Dust opacity file names'],
	['gsmin','1e-5', 'Size (in cm) of smallest dust species'],
	['gsmax','0.1', 'Size (in cm) of largest dust species'],
	['gsdist_powex','-3.5','Grain Distribution power index'],
	['lnk_fname','["/home/cas213/project_len/jena/astrosil_WD2001_new_sorted.lnk"]','Complex Opacity File Name'],
	['mixabun','[1]','Mass fractions of gas componenets'],
        ['dusttogas', '0.01', ' Dust-to-gas mass ratio'],
	['filepath', "'/home/cas213/project_lent/hltau8_0.05_0.0001'", 'Fargo Filepath'],
        ['Rpl', '[30.0*au]', 'Orbital Radius of Planet']
	['fargo_nicell_exclude', '10', 'Number of inner radial cells to exclude from Fargo Data']]

    return defpar


def getDustDensity(grid=None, ppar=None):
    """Calculates the dust density distribution in a protoplanetary disk.
   
    Parameters
    ----------
    grid : radmc3dGrid
           An instance of the radmc3dGrid class containing the spatial and frequency/wavelength grid
    
    ppar : dictionary
           A dictionary containing all parameters of the model
    
    Returns
    -------
    Returns the volume density in g/cm^3
    """
    
    #au = 14959787070000.
    rr, th = np.meshgrid(grid.x, grid.y)
    zz = rr * np.cos(th)
    rcyl = rr * np.sin(th)

    # Calculate the pressure scale height as a function of r, phi
    hp = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    dum = ppar['hrdisk'] * (rcyl/ppar['hrpivot'])**ppar['plh'] * rcyl
    dum = dum.swapaxes(0, 1)
    for iz in range(grid.nz):
        hp[:, :, iz] = dum

    pth = ppar['filepath'] 
    d = fp.fargo3d()
    d.readDims(path=pth)
    d.readGrid(path=pth)
    d.readData(path=pth, frame=600, ddens=True, gdens=True)

    rc_dum = d.rc*ppar['Rpl']

    ixmin = ppar['fargo_nicell_exclude']
    iix = ((grid.x>=rc_dum[ixmin:].min())&(grid.x<=rc_dum[ixmin:].max()))  
    d_x = grid.x[iix]
    imin = iix.min()

    d.isigmagas = np.zeros([grid.nz,grid.nx], dtype=float)
    sigmagas_sp = itp.interp2d(rc_dum[ixmin:], d.phic+np.pi, d.sigmagas.T[:,ixmin:], kind='linear', bounds_error=False, fill_value=np.NaN)
    d.isigmagas[:,iix] = sigmagas_sp(d_x, grid.z)
    

    print(rc_dum[ixmin:].min() / au, rc_dum[ixmin:].max() / au) 
    ii = (grid.x<=rc_dum[ixmin:].min())
    if ii.__contains__(True):
	ii_min = ii.argmin()
	sig0 = d.isigmagas[:,ii_min].mean()
	print(sig0)
	for iz in range(grid.nz):
	    d.isigmagas[iz,ii] = sig0*(grid.x[ii]/grid.x[ii_min])**ppar['plsig1']
    ii = (grid.x>=rc_dum[ixmin:].max())
    if ii.__contains__(True):
	ii_max = ii.argmax()-1
	sig0 = d.isigmagas[:,ii_max].mean()
	print(sig0)
	for iz in range(grid.nz):
	    d.isigmagas[iz,ii] = sig0*(grid.x[ii]/grid.x[ii_max])**ppar['plsig1']

    gasdens = d.isigmagas.T
    z0 = np.zeros([grid.nx, grid.nz, grid.ny], dtype=np.float64)
    rhogas = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    for iz in range(grid.nz):
        for iy in range(grid.ny):
	    for ix in range(grid.nx):
            	rhogas[ix, iy, iz] = gasdens[ix,iz] / (hp[ix, iy, iz]) * np.exp(-0.5 * ((zz[iy, ix])-z0[ix,iz,iy])*((zz[iy, ix])-z0[ix,iz,iy])/(hp[ix, iy, iz]*hp[ix, iy, iz]))
    rhogas = rhogas/np.sqrt(2.0*np.pi) + ppar['bgdens']

    if 'mdisk' in ppar:
        if ppar['mdisk'] != 0.:
            # Calculate the volume of each grid cell
            vol = grid.getCellVolume()
            mass = (rhogas * vol).sum(0).sum(0).sum(0)
	    fraction = 0.99*ppar['mdisk']/mass
            rhogas = rhogas * fraction
    d.isigmagas = d.isigmagas*fraction
    
    d.isigmadust = []

    for i in range(d.Ndust):
	dum1 = np.zeros([grid.nz,grid.nx],dtype=float)
	dust = d.sigmadust[i].T
	sigmadust_sp = itp.interp2d(rc_dum[ixmin:], d.phic+np.pi, dust[:,ixmin:], kind='linear', bounds_error=False, fill_value=np.NaN)
	dum1[:,iix] = sigmadust_sp(d_x,grid.z)

	
    	ii = (grid.x<=rc_dum[ixmin:].min())
    	if ii.__contains__(True):
	    ii_min = ii.argmin()
	    sig0 = dum1[:,ii_min].mean()
	    print(sig0)
	    for iz in range(grid.nz):
	    	dum1[iz,ii] = sig0*(grid.x[ii]/grid.x[ii_min])**ppar['plsig1']

	ii = (grid.x>=rc_dum[ixmin:].max())
    	if ii.__contains__(True):
	    ii_max = ii.argmax()-1
	    sig0 = dum1[:,ii_max].mean()
	    print(sig0)
	    for iz in range(grid.nz):
	    	dum1[iz,ii] = sig0*(grid.x[ii]/grid.x[ii_max])**ppar['plsig1']

	d.isigmadust.append(dum1)
    if ppar['ngs']>1:
	gmin = ppar['gsmin']
	gmax = ppar['gsmax']
	ngs = float(ppar['ngs'])
	gdens = 3.6
        gs = gmin * (gmax/gmin)**(np.arange(ngs,dtype=np.float64)/(ngs-1))
        N0 = (3/(8*np.pi*gdens*np.sqrt(gmax))) * ppar['dusttogas'] * ppar['mdisk']
        mbin = np.zeros([int(ngs)],dtype = np.float64)
	for i in range(len(gs)):
    	    mbin[i] = 8*np.pi*gdens*N0*(gs[i]**0.5)/3

	massfrac = mbin/mbin.sum(0)
	
    dust0 = d.isigmadust[0]
    dust1 = d.isigmadust[1]
    dust2 = d.isigmadust[2]
    dust3 = d.isigmadust[3]
    dust4 = d.isigmadust[4]
    
    print(dust0.shape)
    print(d.isigmagas.shape)
    ynew = np.zeros([grid.nx,grid.nz,ngs],dtype=np.float64)
    for iz in range(grid.nz):
	for ix in range(grid.nx):
	    sigg = d.isigmagas[iz,ix]
	    #if sigg == 0.0:
		#xnew = np.array([5.,5.,5.,5.,5.,5.,5.,5.,5.,5.])
	    #else:
	    xnew = (np.pi/2.) *gs * gdens / sigg
 	    x = np.array(d.stokesNr)
  	    y = np.array([dust0[iz,ix],dust1[iz,ix],dust2[iz,ix],dust3[iz,ix],dust4[iz,ix]])
	    for gsi in range(len(gs)):
		if xnew[gsi] < min(x):
		    ynew[ix,iz,gsi] = sigg
		else:
	    	    sts = itp.interp1d(x,y, kind='linear', bounds_error=False, fill_value=dust4[iz,ix])
		    ydum = sts(xnew)
		    ynew[ix,iz,gsi] = float(ydum[gsi])

    rho = np.zeros([grid.nx, grid.ny, grid.nz, int(ngs)], dtype=np.float64)
    z0 = np.zeros([grid.nx, grid.nz, grid.ny], dtype=np.float64)

    for igs in range(len(gs)):
    	for iz in range(grid.nz):
            for iy in range(grid.ny):
	    	for ix in range(grid.nx):
            	    rho[ix, iy, iz, igs] = ynew[ix,iz,igs]/(hp[ix, iy, iz])*np.exp(-0.5*((zz[iy, ix])-z0[ix,iz,iy])*((zz[iy, ix])-z0[ix,iz,iy])/(hp[ix, iy, iz]*hp[ix, iy, iz]))
    rho = rho/np.sqrt(2.0*np.pi) + ppar['bgdens']

    for igs in range(len(gs)):
	rho[:,:,:,igs] = rho[:,:,:,igs]*massfrac[igs]

    if 'mdisk' in ppar:
        if ppar['mdisk'] != 0.:
	    md = np.zeros([len(gs)], dtype=np.float64)
            # Calculate the volume of each grid cell
	    for igs in range(len(gs)):
            	vol = grid.getCellVolume()
		rho_d = rho[:,:,:,igs]
		x = np.multiply(rho_d,vol)
            	md[igs] = np.sum(x)
	    mass = md.sum(0)
            rho = rho * (0.01*ppar['mdisk'] / mass)

    return rho

def getGasDensity(grid=None, ppar=None):
    """Calculates the gas density distribution in a protoplanetary disk.
    
    Parameters
    ----------
    grid : radmc3dGrid
           An instance of the radmc3dGrid class containing the spatial and frequency/wavelength grid
    
    ppar : dictionary
           A dictionary containing all parameters of the model
    
    Returns
    -------
    Returns the volume density in g/cm^3
    """
    
    rr, th = np.meshgrid(grid.x, grid.y)
    zz = rr * np.cos(th)
    rcyl = rr * np.sin(th)

    # Calculate the pressure scale height as a function of r, phi
    hp = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    dum = ppar['hrdisk'] * (rcyl/ppar['hrpivot'])**ppar['plh'] * rcyl
    dum = dum.swapaxes(0, 1)
    for iz in range(grid.nz):
        hp[:, :, iz] = dum

    pth = ppar['filepath']
    d = fp.fargo3d()
    d.readDims(path=pth)
    d.readGrid(path=pth)
    d.readData(path=pth, frame=600, gdens=True)
    rc_dum = d.rc*ppar['Rpl']

    ixmin = ppar['fargo_nicell_exclude']
    x_min_bound = rc_dum[ixmin:].min()
    x_max_bound = rc_dum[ixmin:].max()
    iix = ( (grid.x>=x_min_bound)&(grid.x<=x_max_bound) )

    sigma_sp = itp.interp2d(rc_dum[ixmin:], d.phic+np.pi, d.sigmagas.T[:,ixmin:], kind='linear', bounds_error=False, fill_value=np.NaN)
    d.isigma = sigma_sp(grid.x[iix], grid.z)
    alpha = d.sigmagas[0,0]*rc_dum[0]
    for iz in range(grid.nz):
        for ix in range(grid.nx):
	    if (np.isnan(d.isigma[iz,ix]) == True):
        	d.isigma[iz,ix] = alpha / grid.x[ix] #THIS NEEDS IMPROVING
    
    gasdens = d.isigma.T
    z0 = np.zeros([grid.nx, grid.nz, grid.ny], dtype=np.float64)
    rho = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    for iz in range(grid.nz):
        for iy in range(grid.ny):
	    for ix in range(grid.nx):
            	rho[ix, iy, iz] = gasdens[ix,iz] / (hp[ix, iy, iz]) * np.exp(-0.5 * ((zz[iy, ix])-z0[ix,iz,iy])*((zz[iy, ix])-z0[ix,iz,iy])/(hp[ix, iy, iz]*hp[ix, iy, iz]))
    rho = rhogas/np.sqrt(2.0*np.pi) + ppar['bgdens']
   
    if 'mdisk' in ppar:
        if ppar['mdisk'] != 0.:
            # Calculate the volume of each grid cell
            vol = grid.getCellVolume()
            mass = (rho * vol).sum(0).sum(0).sum(0)
	    fraction = 0.99*ppar['mdisk']/mass
            rho = rho * fraction
    return rho

def getGasAbundance(grid=None, ppar=None, ispec=''):
    """Calculates the molecular abundance. 
    
    The number density of a molecule is rhogas * abun 
   
    Parameters
    ----------
    grid  : radmc3dGrid
            An instance of the radmc3dGrid class containing the spatial and wavelength grid
    
    ppar  : dictionary
            Dictionary containing all parameters of the model 
    
    ispec : str
            The name of the gas species whose abundance should be calculated

    Returns
    -------
    Returns an ndarray containing the molecular abundance at each grid point
    """

    # Read the dust density and temperature
    try: 
        data = analyze.readData(ddens=True, dtemp=True, binary=True)
    except:
        try: 
            data = analyze.readData(ddens=True, dtemp=True, binary=False)
        except:
            raise RuntimeError('Gas abundance cannot be calculated as the required dust density and/or temperature '
                               + 'could not be read in binary or in formatted ascii format.')

    # Calculate continuum optical depth 
    data.getTau(axis='xy', wav=0.55)
    
    nspec = len(ppar['gasspec_mol_name'])
    if ppar['gasspec_mol_name'].__contains__(ispec):

        sid = ppar['gasspec_mol_name'].index(ispec)
        # Check where the radial and vertical optical depth is below unity
        gasabun = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)  
        
        for spec in range(nspec):
            gasabun[:, :, :] = ppar['gasspec_mol_abun'][sid]
           
        for iz in range(data.grid.nz):
            for iy in range(data.grid.ny):
                ii = (data.taux[:, iy, iz] < ppar['gasspec_mol_dissoc_taulim'][sid])
                gasabun[ii, iy, iz] = 1e-90

                ii = (data.dusttemp[:, iy, iz, 0] < ppar['gasspec_mol_freezeout_temp'][sid])
                gasabun[ii, iy, iz] = ppar['gasspec_mol_abun'][sid] * ppar['gasspec_mol_freezeout_dfact'][sid]

    else:
        gasabun = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) + 1e-10
        txt = 'Molecule name "'+ispec+'" is not found in gasspec_mol_name \n A default 1e-10 abundance will be used'
        warnings.warn(txt, RuntimeWarning)

    # gasabun = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    # gasabun[:,:,:] = ppar['gasspec_mol_abun'][0] / (2.4*mp)

    return gasabun


def getVTurb(grid=None, ppar=None):
    """Calculates the turbulent velocity field
    
    Parameters
    ----------
    grid : radmc3dGrid
           An instance of the radmc3dGrid class containing the spatial and frequency/wavelength grid
    
    ppar : dictionary
           A dictionary containing all parameters of the model
    
    Returns
    -------
    Returns an ndarray with the turbulent velocity in cm/s
    """

    vturb = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64) + ppar['gasspec_vturb']
    return vturb


def getVelocity(grid=None, ppar=None):
    """Calculates the velocity field in a protoplanetary disk.
    
    Parameters
    ----------
    grid : radmc3dGrid
           An instance of the radmc3dGrid class containing the spatial and frequency/wavelength grid
    
    ppar : dictionary
           A dictionary containing all parameters of the model
    
    Returns
    -------
    Returns the gas velocity in cm/s
    """

    nr = grid.nx
    nphi = grid.nz
    nz = grid.ny
    rcyl = grid.x

    vel = np.zeros([nr, nz, nphi, 3], dtype=np.float64)
    vkep = np.sqrt(gg * ppar['mstar'][0]/rcyl)
    for iz in range(nz):
        for ip in range(nphi):
            vel[:, iz, ip, 2] = vkep

    return vel
