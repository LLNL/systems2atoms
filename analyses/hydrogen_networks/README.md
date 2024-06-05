# S2A System-Scale Analysis: Hydrogen Delivery Costs

## Description
This folder contains the scripts to run the system-scale analysis and make the plots for the manuscript "Systems-to-Atoms Examination of Formic Acid as a Liquid Organic Hydrogen Carrier in Transportation Applications".
Input files, raw output files, and plots are also included.  

* `cross-scale data handoff`: Outputs from material and component modeling that are passed to the system scale and/or used to make plots.
* `inputs`: Input files to run the system-scale analysis.
* `outputs`: Folders containing raw output files.
* `plots`: Plots and scripts used to make these plots.
* `requirements.txt`: List of Python packages needed to run the system-scale analysis.
* `s2a_sys.yml`: Environment file.
* `s2a_systems_functions.py`: Functions, including mass and energy balances, equipment sizing and costing, levelized cost of hydrogen for hydrogen delivery via compressed hydrogen, liquid hydrogen, and formic acid.
* `s2a_systems_run_analysis.ipynb`: Script used to run the system-scale analysis for user-specified inputs (e.g., in an `input params_[scenario group].xlsx` file). Imports `s2a_systems_functions.py`.

## Getting Started
### [For first-time users] Create the conda environment
* Open Anaconda Prompt.
* Nagivate to the main folder.
* Type:
```
conda env create -f s2a_sys.yml
```

### Activate the conda environment
* Open Anaconda Prompt.
* Nagivate to the main folder.
* Type:
```
conda activate s2a_sys
```

### Run the analysis
* Open `s2a_systems_run_analysis.ipynb`. [Recommended: open in Jupyter Notebook.]
* Run `s2a_systems_run_analysis.ipynb`.
  * This will run the "baseline" scenarios (with input parameters specified in `input params_baseline.xlsx`) and create an `outputs [today's date] baseline` folder containing raw output files in the `outputs` folder.
* [Optional] To run a different set of scenarios, change the input filename and output folder name (if desired) under the `USER INPUT` section:
```
# specify input parameter file
df_input_params_all = pd.read_excel('inputs/[your input filename].xlsx')

# specify output folder
output_folder = 'outputs/[your output folder name]'
```
Note: `[your input filename].xlsx` needs to exist in the `inputs` folder. See instructions below for creating new input parameter files.

### [Optional] Add or update scenarios and input parameters
* Navigate to the `inputs` folder.
* Add an `.xlsx` file using the same format as one of the `input params_[scenario group].xlsx` files.
  * The easiest way to do this would be to copy an existing `input params_[scenario group].xlsx` file and make changes in the copied file.
  * Notes on the input parameters can be found in the `input params_[scenario group].xlsx` files.
* [Not recommended] Alternatively, can make changes in one of the existing `input params_[scenario group].xlsx` files.
  * The existing `input params_[scenario group].xlsx` files are used for the analysis in the manuscript. It is recommended to keep these files for reproducibility.
 
## Authors

Mengyao Yuan (yuan13@llnl.gov)  

Bo-Xun Wang (University of Wisconsin-Madison), Corey Myers (LLNL), Thomas Moore (Queensland University of Technology), and Wenqin Li (LLNL) have contributed to the development of the systems model and analysis.

## Citation
[TBD]
