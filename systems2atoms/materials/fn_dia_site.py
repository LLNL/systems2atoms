import numpy as np
import matplotlib.pyplot as plt
import sys
from ase.cluster import wulff_construction
from ase import Atoms
from ase.build import bulk, fcc111, fcc100, fcc211, add_adsorbate
from ase.io import read, write
from ase.constraints import FixAtoms
import subprocess
from NN_list_fn import *

plt.rcParams["font.size"]=14

cn_dict=[]
# max_particle_dia=10
def fn_dia_site(particle_dia):
    ###Diameter estimation

    # particle_dia=particle_dia#sys.argv[1] #nm
    #density of Pd = 12.023 g/cm3

    particle_dia=float(particle_dia)*1e-9 #(~1 nm)#density of Pd = 12.023 g/cm3
    surface_area_particle=3.14*particle_dia**2 #pi*diameter^2
    particle_weight=12.023*3.14*particle_dia**3*1e6/6 #pi*dia^3/6 #grams

    ##Molar mass of Pd = 106.42 g/mol
    Avogadro_Num=6.023*1e23

    num_atoms=particle_weight/106.42*Avogadro_Num

    # print(i, "number of atoms = ", num_atoms)


    a = 200  # cube size
    cell = [a, a, a]  # cubic cell
    #atoms = bulk('Cu', 'fcc', a=a)  # create a copper crystal
    #atoms += water  # add the water molecule to the copper crystal
        

    surfaces = [(1,0,0),
        (1,1,0),
        (1,1,1),
        (2,1,0),
        (2,1,1),
        (2,2,1),
        (3,1,0),
        (3,1,1),
        (3,2,0),
        (3,2,1),
        (3,2,2),
        (3,3,1),
        (3,3,2)]
    esurf = [
    1.52,
    1.57,
    1.36,
    1.63,
    1.61,
    1.50,
    1.63,
    1.57,
    1.64,
    1.59,
    1.45,
    1.53,
    1.46]        # Surface energies.
    lc = 3.89000
    size = int(num_atoms)  # Number of atoms
    atoms = wulff_construction('Pd', surfaces, esurf,
                            size, 'fcc',
                            rounding='above', latticeconstant=lc)


    atoms.set_cell(cell)  # set the unit cell size
    atoms.center()  # place the molecule at the center of the unit cell
    #outname = "".join([mol[i],'_POSCAR'])
    #write(outname, atoms, format='vasp',direct=True)
    write(f"Pd_wulff_POSCAR_{np.round(particle_dia/1e-09, 2)}_nm",atoms, format='vasp',direct=True)


    # subprocess.run(["python3 NN_list.py Pd_wulff_POSCAR"], shell=True))

    cn_dict.append(NN_list_fn(f"Pd_wulff_POSCAR_{np.round(particle_dia/1e-09, 2)}_nm"))  ##rectify the 10^-9 being written

    # print(cn_dict)

 #####Surface atom population######

 # Create a list of dictionaries (cn_dict) - replace with your data
 # cn_dict = [{5: 24, 6: 72, 7: 24, 8: 6, 9: 48, 10: 24, 11: 12, 12: 1}, {5: 10, 6: 20, 7: 30, 8: 40, 9: 50, 10: 60, 11: 70}]

 # Initialize lists to store data for plotting
    x_values = []  # x-axis values
    y_values_100 = []  # y-axis values for "100"
    y_values_111 = []  # y-axis values for "111"
    y_values_corner = []  # y-axis values for "corner"
    y_values_edge = []  # y-axis values for "edge"

    for i, data in enumerate(cn_dict):
        sum_surface = sum(data.get(key, 0) for key in range(5, 12))
        
        # Calculate percentages for each category
        percentage_100 = (data.get(8, 0) / sum_surface) * 100
        percentage_111 = (data.get(9, 0) / sum_surface) * 100
        percentage_corner = (data.get(5, 0) / sum_surface) * 100
        percentage_edge = (data.get(7, 0) / sum_surface) * 100
        
        # Append values to lists for plotting
        # x_values.append(i + 1)
        # y_values_100.append(percentage_100)
        # y_values_111.append(percentage_111)
        # y_values_corner.append(percentage_corner)
        # y_values_edge.append(percentage_edge)

    # Create subplots
    # fig, ax = plt.subplots()

    # Plot the data
    # ax.plot(x_values, y_values_100, marker='o', linestyle='-', label="100")
    # ax.plot(x_values, y_values_111, marker='o', linestyle='-', label="111")
    # ax.plot(x_values, y_values_corner, marker='o', linestyle='-', label="corner")
    # ax.plot(x_values, y_values_edge, marker='o', linestyle='-', label="edge")

    # Show the legend
    # ax.legend()

    # Show the plot
    # ax.tick_params(axis="both",direction="out",labelsize=12)
    # ax.set_ylabel('Surface Atom Population (%)', fontweight="bold")#, fontsize=18)
    # ax.set_xlabel('Pd particle diameter (nm)', fontweight="bold")#, fontsize=18)
    # plt.xticks(weight = 'bold')
    # plt.yticks(weight = 'bold')
    # plt.savefig('dia_site.jpg',dpi=330,bbox_inches='tight')
    # plt.show()
    return percentage_100, percentage_111, percentage_corner, percentage_edge
