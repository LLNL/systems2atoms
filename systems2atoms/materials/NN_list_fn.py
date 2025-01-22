#Depends on ASAP3 - check out https://wiki.fysik.dtu.dk/asap/Neighbor%20lists
#By Joakim Halldin Stenlid SUNCAT center, SLAC/Stanford Uni 2020
# syntax python2 structure-file
# make sure to set equillibrium distance for metal in your system!
from ase import Atoms
from ase import geometry
from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
import sys, math
from ase import neighborlist
from asap3 import *


# loop over all atoms
def NN_list_fn(infile):
    #INPUT
    #infile=sys.argv[1]          # Name of structure input file
    eq_dist=2.772               # Metal equilibrium distance in favored coordination environment, 2.772 for Pt
    tol=0.1                     # Max tolerated bond strain. Note: strain model not tested beyond 8%
    eq_dist_tol=eq_dist*(1+tol) # Max bond distance to be considered a bond
    #read structure and basic analysis
    atoms=read(infile)
    N = atoms.get_number_of_atoms()
    nblist = FullNeighborList(eq_dist_tol, atoms, driftfactor=0.05)
    pos=atoms.get_positions()
    cell=atoms.get_cell()
    elements=atoms.get_chemical_symbols()
    cn_site_record=[]
    for i in range(len(atoms)):
        nb, dvec, d2 = nblist.get_neighbors(i) #neigborlist, pairwise distance vector, and squared norm of distance vectors
    #    print("\n", nb,"for atom",i)
        #print(dvec,"for atom",i)
        cn_site = len(nb)
    # print(cn_site)
        cn_nb = []
        alpha = np.zeros(12,dtype='int') #change to 12
        alpha[np.arange(cn_site)] = alpha[np.arange(cn_site)] + 1
        for ii in range(cn_site):
            neighbor = nb[ii]
            cn_neighbor = len(nblist.get_neighbors(neighbor)[0])
            cn_nb.append(cn_neighbor)
            alpha[cn_neighbor-1] = alpha[cn_neighbor-1] + 1
        alpha = np.delete(alpha,[1,2])
        # print(i, cn_site, alpha)
        cn_site_record.append(cn_site)
    #print(cn_site_record)
    unique, counts = np.unique(cn_site_record, return_counts=True)

    # print("\n", dict(zip(unique, counts)))
    return dict(zip(unique, counts))
