# TOF Analysis for Formic Acid Dehydrogenation on Pd Nanoparticles

## Description

This repository contains scripts to compute **Turnover Frequency (TOF)** data (mol H<sub>2</sub>/mol catalyst·hr) for Palladium nanoparticles of specified sizes over a range of temperatures and pressures. The main script, `tof_map_dia.py`, calculates TOF values and generates 2D heatmaps (TOF vs. T and P) for visualization.

The workflow uses **CATMAP** for microkinetics modeling (MKM), Wulff construction for nanoparticle geometry, and additional Python scripts for data processing. For detailed information about CATMAP, refer to its [documentation](https://catmap.readthedocs.io/en/latest/index.html).

### Key Features
- Flexible computation of TOF for Pd nanoparticles.
- Adjustable temperature and pressure ranges.
- Outputs include raw TOF data and heatmaps.

---

## File Structure and Workflow

### 1. CATMAP Files
- **Input Files**:
  - `HCOOH_decomposition_211.mkm`, `HCOOH_decomposition_111.mkm`: Define microkinetics models for Pd facets.
  - `energies_111.txt`, `energies_211.txt`: Literature data for formic acid dehydrogenation kinetics ([source](https://pubs.acs.org/doi/10.1021/cs400664z)).

- **Run Scripts**:
  - `mkm_job_111.py`, `mkm_job_211.py`: Execute MKM simulations.

- **Output Files**:
  - `.pkl` files (e.g., `HCOOH_decomposition_111.pkl`): Contain MKM results.
  - `.log` files: Logs of MKM simulations.
  - PDF files (e.g., `turnover_frequency.pdf`): Visualizations of TOF data.

---

### 2. Wulff Construction Files
- **Input Scripts**:
  - `fn_dia_site.py`: Constructs Wulff particles based on surface energies ([https://crystalium.materialsvirtuallab.org/](https://crystalium.materialsvirtuallab.org/)).
  - `NN_list_fn.py`: Calculates the surface fraction for different facets.

- **Output**:
  - Geometry files (e.g., `Pd_wulff_POSCAR_4.0_nm`): Atomic coordinates for the Wulff particle.

---

### 3. Supporting Scripts
- **`get_tof.py`**: Converts MKM rate data to TOF by considering catalyst weight and surface area.
- **`parse_data_def_T_P_stored.py` / `parse_data_def_T_P.py`**: Uses Arrhenius fitting to derive activation barriers ([details](https://en.wikipedia.org/wiki/Arrhenius_equation)).

---

## Getting Started

### 1. Set Up the Environment
1. Install [Anaconda](https://www.anaconda.com/).
2. Navigate to the project directory using the Anaconda Prompt.
3. Create and activate the environment:
   ```bash
   conda env create -f s2a_tof.yml
   conda activate s2a_tof

### 2. Run the TOF Analysis
To compute TOF, run the following command:
```
python tof_map_dia.py <Particle_diameter (nm)>
```
Replace <Particle_diameter (nm)> with the desired nanoparticle size in nanometers.

## Customization
### Adjust Temperature and Pressure Ranges
To customize the temperature (`T`) and pressure (`P`) ranges:
1. Open the `tof_map_dia.py` script.
2. Locate the **USER INPUT** section:
```python
# User inputs for Temperature and Pressure Range
T = np.linspace(300, 400, 20)  # Temperature range (K)
P = np.linspace(1, 100, 20)    # Pressure range (atm)
```
3. Modify the ranges as needed.
4. Ensure these ranges match the descriptor_ranges in .mkm files: 
  
``` python
descriptor_ranges = [[300, 400], [1, 100.0]]
resolution = 20
```

## Output

All results are saved in the **output** directory, including:
- `TOF Data: .csv, .xlsx, .npz files.`

- `TOF Heatmaps: .jpg images.`

## Requirements

To run the project, you need the following tools and libraries:

1. [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/index.html):
```
conda install -c conda-forge ase
```
2. [CATMAP](https://catmap.readthedocs.io/en/latest/index.html).
3. Python Libraries:
	- `scipy`
	- `matplotlib`
	- `pandas`
	- `openpyxl`
4. [ASAP3](https://asap3.readthedocs.io/en/latest/):
	Used for neighbor list generation in Wulff construction.

## References
- [CATMAP](https://catmap.readthedocs.io/en/latest/index.html)
- Kinetic Data Source: [DOI: 10.1021/cs400664z](https://pubs.acs.org/doi/10.1021/cs400664z)

## Authors

Wenyu Sun (sun39@llnl.gov)  

Shyam Deo (deo4@llnl.gov)

## Citation
For citation, refer to:
[[DOI: 10.1021/acs.iecr.4c03344](https://doi.org/10.1021/acs.iecr.4c03344)]
