import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=4, suppress=True)


##INPUTS
def get_tof(system, area_fraction, particle_dia, rate): #rate=from rate_T_P.py script
    # system=sys.argv[1] #111, 100 or 211
    # particle_dia=float(sys.argv[2])  #nm
    particle_dia=float(particle_dia)*1e-9 #(~1 nm)#density of Pd = 12.023 g/cm3
    surface_area_particle=3.14*particle_dia**2 #pi*diameter^2
    particle_weight=12.023*3.14*particle_dia**3*1e6/6 #pi*dia^3/6

    #print("weight of Pd (g) per nanocube = ", 12.023*size_particle_cube**3*1e6)
    ##Molar mass of Pd = 106.42 g/mol
    Avogadro_Num=6.023*1e23

    #111 data
    def get_sites_per_area(system):
        num_sites_per_area=None
        if system=="111":
            num_sites_per_area=1/(6.552391506*9*1e-20) #double-check #per m2 ##1 per 59.88 A2##from MKM paper - 1 per 3x3 Pd111 (3.92*3.92) - srt(3)/4*3.92*3.92*9

        elif system=="100":
        #100 data
            num_sites_per_area=1/(7.56605*9*1e-20) #taken from my stanford data #taken from ASE GUI structure here=5.544*5.544 -~/work1/carriers/wsu/scripts

        elif system=="211":
        #211 data
            num_sites_per_area=1/(6.17672106*9*1e-20) #6.465839657 (1 atom) = taken from my stanford data - using cos theta between 111 and 211 #391.2
        return num_sites_per_area

    #TOF (mol H2/mol Pd*sec)
    print(get_sites_per_area(system))
    TOF=rate*get_sites_per_area(system)*area_fraction*surface_area_particle*106.42/((particle_weight)*Avogadro_Num) #Avogadro_Num is to convert H2 molecules produced per site into moles

    # print("moles of Pd per cube = ", 12.023*size_particle_cube**3*1e6/106.42)
    print("TOF (mol H2/mol Pd*sec) = ", TOF)
    print("TOF (mol H2/mol Pd*hr) = ", TOF*3600)
    # rate_dehydg_per_particle=P*A_FA*np.exp(-Ea_FA/(R*T))*num_sites_per_area*surface_area_particle/Avogadro_Num#rate expression.
    return TOF*3600 #TOF (mol H2/mol Pd*hr)

print(get_tof("111", 0.2, 5, 0.02))
print(get_tof("100", 0.2, 5, 0.02))
print(get_tof("211", 0.2, 5, 200))
