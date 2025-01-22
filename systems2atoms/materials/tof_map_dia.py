import numpy as np
import matplotlib.pyplot as plt
# from rate_T_P import *
from get_tof import *
from fn_dia_site import *
from parse_data_def_T_P_stored import provide_rate_data
import sys
import subprocess
import csv
import os

#usage python3 tof_map_dia.py particle_dia(nm)
#Note - Please initiate same range for T,P in the HCOOH_decomposition_111.mkm and HCOOH_decomposition_211.mkm files as desired here in the TOF map
particle_dia=float(sys.argv[1]) 

# Create an output directory
dir_name = "output"

try:
    os.mkdir(dir_name)
    print(f"Directory '{dir_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{dir_name}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

print("##############STOP####################")
print("\nDid you initiate same range for P in the HCOOH_decomposition_111.mkm and HCOOH_decomposition_211.mkm files as desired here in the TOF map?")
# Generate random data for the heatmap (replace with your data) #T and P
# T=float(sys.argv[2])#400 #K
# P=sys.argv[3]#1.1 #atm
#User inputs on Temperature and Pressure Range -
T = np.linspace(300, 400, 20)  #kelvin #User input
P = np.linspace(1, 100, 20)  #atm #initiate same range for P in the mkm file as here


#percentage_100, percentage_111, percentage_corner, percentage_edge (from fn_dia_site)
values_111 = fn_dia_site(particle_dia)[1]/100#np.linspace(0, 0.5, 20)  
values_211 = fn_dia_site(particle_dia)[3]/100 #for now MKM taken as same as 111 
values_100 = fn_dia_site(particle_dia)[0]/100  


# Create a grid of (111_values, 100_values)
T_value, P_value=np.meshgrid(T, P)


# print(np.meshgrid(T, P))

# Values_111, Values_211, Values_100 = np.meshgrid(values_111, values_211, values_100)
# Values_100=1-Values_111-Values_211

###Run and store the arrhenius plots for the range of T and P
##system=="111" or system=="100" 
subprocess.run(["python3 mkm_job_111.py"], shell=True)

subprocess.run(["python3 log_analysis.py production_rate HCOOH_decomposition_111.log"], shell=True)

# fit_results[pressure] = {'a_fit': a_fit, 'b_fit': b_fit}
# return fit_results ##store these for several P and access for new rates

fit_results_111=provide_rate_data("production_rate_table.txt") #T range is 298-420 K.
fit_results_100=fit_results_111

##system=="211"
subprocess.run(["python3 mkm_job_211.py"], shell=True)

subprocess.run(["python3 log_analysis.py production_rate HCOOH_decomposition_211.log"], shell=True)

fit_results_211=provide_rate_data("production_rate_table.txt") #T range is 298-420 K.

def rate_T_P(system, T, P):
    if system=="111" or system=="100" : #for now 100 is approximated as using 111 data    
        m=fit_results_111[P]["a_fit"]
        c=fit_results_111[P]["b_fit"]
        rate=np.exp(m*1/T+c)
        # print(rate)

    elif system=="211":
        m=fit_results_211[P]["a_fit"]
        c=fit_results_211[P]["b_fit"]
        rate=np.exp(m*1/T+c)
    return rate

# Loop through individual P_value values and calculate rate_111
data = np.zeros_like(T_value)

for i in range(P_value.shape[0]):
    for j in range(P_value.shape[1]):
        individual_P_value = P_value[i, j]
        print(T_value[i, j], individual_P_value)
        # rate_111_value = rate_T_P("111", T_value[i, j], individual_P_value)
        rate_111=rate_T_P("111", T_value[i, j], individual_P_value)
        rate_100=rate_T_P("111", T_value[i, j], individual_P_value)#rate_111 #put same as 111 so that this additional calculation is skipped
        rate_211=rate_T_P("211", T_value[i, j], individual_P_value)

        data[i,j]=get_tof("111",values_111,particle_dia,rate_111)+get_tof("211",values_211,particle_dia,rate_211)+get_tof("100",values_100,particle_dia,rate_100) #get_tof(system, area_fraction, particle_dia, rate): #rate=from rate_T_P.py script
        
# Create the heatmap
plt.figure(figsize=(8, 6))
# plt.imshow(data, cmap='viridis', extent=[T_value.min(), T_value.max(), P_value.min(), P_value.max()], origin='lower', aspect="equal")
plt.imshow(data, cmap='viridis', extent=[T.min(), T.max(), P.min(), P.max()], origin='lower')
plt.colorbar(label='TOF(mol H2/mol Pd*hr)')


# Save the data as a CSV file
with open(f'output/data_{particle_dia}_nm.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in data:
        csvwriter.writerow(row)


# Save the data, T_value, and P_value in a single .npz file
np.savez(f'output/data_and_grid_{particle_dia}_nm.npz', data=data, T_value=T_value, P_value=P_value)

import pandas as pd

# Assuming you have the data, T_value, and P_value arrays defined
# Create a DataFrame
df = pd.DataFrame({'T_value (K)': T_value.flatten(), 'P_value (atm)': P_value.flatten(), 'TOF_Data (mol H2/mol Pd*hr)': data.flatten()})

# Save the DataFrame to an Excel file
df.to_excel(f'output/Pd_data_{particle_dia}_nm.xlsx', index=False)

# Now you have an 'data.xlsx' Excel file that contains the data, T_value, and P_value


# Set axis labels and title
plt.xlabel('Temperature (kelvin)')
plt.ylabel('Pressure (atm)')
plt.title(f'Heatmap at {particle_dia} nm')
plt.savefig(f'output/tof_map_{particle_dia}_nm.jpg',dpi=330,bbox_inches='tight')

# Show the plot
# plt.show()


