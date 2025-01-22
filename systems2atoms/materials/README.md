This folder contains the scripts to generate Turnover Frequency (TOF) for Pd at required nanoparticle size across a range of Temperature and pressures. *_111.mkm and *_211.mkm files contain literature data for kinetics of formic acid dehydrogenation on Pd (https://pubs.acs.org/doi/10.1021/cs400664z)

#Run the analysis
Usage - Python tof_map_dia.py Particle_diameter(nm)
#Note - Please initiate same range for T,P in the HCOOH_decomposition_111.mkm and HCOOH_decomposition_211.mkm files as desired here in the TOF map.

Output - TOF data (vs T, P) and the corresponding 2D-heatmap for TOF are written to "output" directory.

Requirements -
	1. ASE - https://wiki.fysik.dtu.dk/ase/about.html 
	2. CATMAP - https://catmap.readthedocs.io/en/latest/index.html
	3. Scipy
	4. Matplotlib
		

