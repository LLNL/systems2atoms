# -*- coding: utf-8 -*-
"""
Functions for S2A systems technoeconomic analysis.

Outputs include .csv file and dataframe containing input parameters and results.

@author: yuan13
"""

#%% IMPORT MODULES

import os
import math
import pandas as pd

import pathlib
this_file = pathlib.Path(__file__).parent.resolve()

#%% INPUT PARAMETERS

# Parameters here are considered somewhat fixed (e.g., tied to design
# decisions or other assumptions). Can move outside if needed. 

# Parameters that are meant to be varied for analysis are in the 
# "input params" Excel files in the "inputs" folder, with default values 
# defined in the technoeconomic analysis function ("calcs") here / below.

# fuel economy (mile/gallon, implied: per vehicle)
truck_fuel_econ_mi_per_gal = 6.0

# truck travel speed (mile/hr)
truck_speed_mi_per_hr = 40.0

# gas truck tube minimum and maximum operating pressures (atm)
# minimum tube pressure = compressed hydrogen refueling station compressor 
# inlet pressure (for *power* calculations)
gas_truck_max_tube_pres_atm = 540.0
gas_truck_min_tube_pres_atm = 50.0

# truck (total) loading and unloading time (hr)
gas_truck_load_unload_time_hr = 3.0
liq_truck_load_unload_time_hr = 6.5

# compressed hydrogen truck (tube trailer) delivered capacity 
# (kg H2/truck-trip)
# 1042 kg corresponds to tube maximum operating pressure of 540 atm
gas_truck_deliv_capacity_kgH2 = 1042.0

# liquid hydrogen truck delivered capacity (kg H2/truck-trip)
liq_truck_deliv_capacity_kgH2 = 3610.0

# liquid truck water volume (m^3/truck-trip)
# corresponds to liquid hydrogen truck deliverd capacity = 3610 kg
# use for calculating LOHC truck delivered capacity
# TODO: use general truck parameters for LOHC
liq_truck_water_vol_cu_m = 56.5

# liquid truck usable delivery capacity (% nominal holding capacity)
liq_truck_usable_capacity_frac = 0.95

# general truck "rated" cargo density (kg/m^3)
# calculated using maximum cargo weight = 25,200 kg and 
# maximum cargo volume = 75 m^3 
# (Nexant et al., "Final Report", "Supplemental Report to Task 2", p.6)
liq_truck_cargo_dens_kg_per_cu_m = 25200.0 / 75.0

# compressed hydrogen terminal inlet hydrogen temperature (deg.C)
# HDSAM V3.1: hydrogen temperature at terminal (low and ambient temperature)
GH2_TML_in_temp_C = 25.0

# LOHC / formic acid terminal inlet hydrogen temperature (deg.C)
# assume same as compressed hydrogen
FAH2_TML_in_temp_C = GH2_TML_in_temp_C

# refueling station hydrogen temperature (deg.C)
# HDSAM V3.1: 40 deg.C = ambient (25 deg.C) + 15 deg.C for 
# "hot soaking condition" = maximum hydrogen temperature at station
# = inlet temperature to compressor and refrigerator
# = outlet temperature from PSA refrigerator (precooling)
STN_H2_temp_C = 40.0

# refueling station hydrogen dispensing temperature (deg.C)
# HDSAM V3.1: maximum dispensing temperature
# = outlet temperature from compressor refrigerator
STN_dispens_temp_C = -40.0

# compressed hydrogen terminal inlet hydrogen pressure (atm) 
# NOTE: hydrogen delivered to compressed hydrogen terminal @ 20 atm 
# in HDSAM V3.1
GH2_TML_in_pres_atm = 20.0

# LOHC / formic acid hydrogen terminal inlet hydrogen pressure (atm)
# assume same as compressed hydrogen (based on Perez-Fortes et al., 2016)
FAH2_TML_in_pres_atm = GH2_TML_in_pres_atm

# compressed hydrogen terminal storage minimum and maximum pressures (atm)
GH2_TML_max_stor_pres_atm = 400.0
GH2_TML_min_stor_pres_atm = 200.0

# liquid hydrogen terminal inlet hydrogen pressure (atm)
LH2_TML_in_pres_atm = 2.0

# liquid hydrogen terminal outlet hydrogen pressure (atm)
# = hydrogen pressure to truck
LH2_TML_out_pres_atm = LH2_TML_in_pres_atm + 1.0

# liquid hydrogen terminal storage boil-off 
# HDSAM V3.1: 0.03% of storage capacity per day at terminal
# use as fraction per day of initial amount of hydrogen stored
LH2_TML_stor_boil_off_frac_per_day = 0.0003

# compressed hydrogen refueling station compressor outlet pressure (bar)
# HDSAM V3.1: main compressor discharge pressure at refueling station for 
# compressed gaseous hydrogen
# different from refueling station compressor discharge pressure for 
# liquid hydrogen
GH2_STN_out_pres_bar =  969.0

# liquid hydrogen refueling station pump inlet pressure (atm)
# HDSAM V3.1: hydrogen supply pressure from dewar
LH2_STN_in_pres_atm = 6.0

# liquid hydrogen refueling station pump outlet pressure (bar) 
# HDSAM V3.1: main compressor discharge pressure at refueling station for 
# liquid hydrogen
# different from refueling station compressor discharge pressure for 
# compressed gaseous hydrogen
LH2_STN_out_pres_bar =  946.0

# LOHC (formic acid) hydrogen refueling station
# dehydrogenation pump inlet pressure (bar)
# = pressure of formic acid in truck
FAH2_STN_dehydr_pump_in_pres_bar = 1.013

# LOHC (formic acid) hydrogen refueling station
# pressure swing adsorption operating pressure (bar)
FAH2_STN_psa_pres_bar = 20.0

# LOHC (formic acid) hydrogen refueling station
# pressure swing adsorption operating temperature (deg.C)
FAH2_STN_psa_temp_C = 25.0

# LOHC (formic acid) hydrogen refueling station compressor outlet 
# pressure (bar)
# = compressor outlet pressure at compressed hydrogen refueling station
# (based on Nexant et al., "Final Report", "Supplemental Report to Task 2", 
# p.18)
FAH2_STN_out_pres_bar = GH2_STN_out_pres_bar

# cascade storage size (% of station capacity) at compressed hydrogen 
# refueling station 
# TODO: consider optimizing cascade size as a fraction of station capacity 
# (optimized in HDSAM V3.1)
GH2_STN_casc_stor_size_frac = 0.25

# cascade storage size (% of station capacity) at liquid hydrogen 
# refueling station 
# see note on cascade storage sizing above
LH2_STN_casc_stor_size_frac = 0.15

# cascade storage size (% of station capacity) at LOHC (formic acid) 
# hydrogen refueling station
FAH2_STN_casc_stor_size_frac  = GH2_STN_casc_stor_size_frac

# discount rate
# real after-tax discount rate on "Financial Inputs" tab in HDSAM V3.1
discount_rate = 0.10

# total tax rate
# total tax rate includes state taxes and federal taxes on "Financial Inputs" 
# tab in HDSAM V3.1
state_taxes = 0.06
federal_taxes = 0.35
total_tax_rate = federal_taxes + state_taxes * (1 - federal_taxes)

# electrolyzer voltage (V) for formic acid production
# hydrogen oxidation: 1.27 V (= 2.5 - 1.23)
# water oxidation: 2.5 V
# references: Li et al., 2021, ACS Sustain. Chem. Eng.; 
# Ramdin et al., 2019, Ind. Eng. Chem. Res.; 
# Crandall et al., 2023, Energy Fuels
electr_volt_V = 1.27
# electr_volt_V = 2.5

# electrolyzer current density (A/m^2) for formic acid production
electr_curr_dens_A_per_sq_m = 2000.0

# ----------------------------------------------------------------------------
# cost levelization

# equipment lifetime (yr)
# TODO: revisit cost allocation; tractor and trailer have different lifetimes 
# in HDSAM V3.1 (5 and 15 years, respectively) 
# TODO: revisit terminal and station reactor and separator (PSA) lifetimes; 
# TODO: revisit terminal vaporizer lifetime
# TODO: revisit terminal electrolyzer lifetime
# for now, use remainder of terminal or station lifetime in HDSAM V3.1
# terminal (formic acid) distillation: Ramdin et al., 2019
TML_compr_life_yr = 15.0
TML_pump_life_yr = 15.0
TML_stor_life_yr = 30.0
TML_react_life_yr = 30.0
TML_vap_life_yr = 30.0
TML_distil_life_yr = 15.0
TML_electr_life_yr = 30.0
liquef_life_yr = 40.0
truck_life_yr = 15.0
STN_compr_life_yr = 10.0
STN_pump_life_yr = 10.0
STN_stor_life_yr = 30.0
STN_refrig_life_yr = 15.0
STN_vap_life_yr = 30.0
STN_react_life_yr = 30.0
STN_psa_life_yr = 30.0

# equipment MACRS deprepreciation schedule length (yr)
# TODO: revisit depreciation length for equipment used for 
# LOHC terminal or refueling station (reactor, catalyst, 
# distillation, electrolyzer, PSA)
# for now, use assumptions in HDSAM V3.1 for similar equipment types
TML_compr_depr_yr = 10.0
TML_pump_depr_yr = 15.0
TML_stor_depr_yr = 15.0
TML_react_depr_yr = 15.0
TML_vap_depr_yr = 15.0
TML_distil_depr_yr = 10.0
TML_electr_depr_yr = 15.0
liquef_depr_yr = 15.0
truck_depr_yr = 5.0
STN_compr_depr_yr = 5.0
STN_pump_depr_yr = 5.0
STN_stor_depr_yr = 5.0
STN_refrig_depr_yr = 5.0
STN_vap_depr_yr = 5.0
STN_react_depr_yr = 15.0
STN_psa_depr_yr = 15.0
FAH2_hydr_catal_depr_yr = 3.0
FAH2_dehydr_catal_depr_yr = 3.0

# MACRS depreciation period table
# read in MACRS depreciation period table (source: HDSAM V3.1)
df_macrs_depr_idx = pd.read_csv(this_file/pathlib.Path('inputs/MACRS depreciation period.csv'))

#%% CONSTANTS AND CONVERSIONS

# gas constant (kJ/kmol-K)
gas_const_kJ_per_kmol_K = 8.3144

# Faraday constant (C/mol)
faraday_const_C_per_mol_e = 9.6485e4

# normal conditions (20 deg.C, 1 atm)
norm_temp_K = 293.15
norm_pres_Pa = 101325.0

# number of electrons transferred in reaction (electrons/mol)
e_per_mol_FA = 2

# molar mass (kg/kmol)
molar_mass_H2_kg_per_kmol = 2.016
molar_mass_CO2_kg_per_kmol = 44.009
molar_mass_FA_kg_per_kmol = 46.025

# density (kg/m^3)
# liquid hydrogen at normal boiling point
dens_liq_H2_kg_per_cu_m = 70.85
# formic acid at ambient conditions (25 deg.C, 1 atm)
dens_FA_kg_per_cu_m = 1220.0

# stoichiometric ratio (mol/mol)
stoic_mol_H2_per_mol_FA = 1.0
stoic_mol_CO2_per_mol_FA = 1.0

# lower heating value (MJ/kg)
low_heat_val_H2_MJ_per_kg = 119.96
low_heat_val_diesel_MJ_per_gal = 135.5

# kWh to MJ (1 kWh = 3.6 MJ)
MJ_per_kWh = 3.6

# celsius to kelvin (K = C + 273.15)
C_to_K = 273.15

# gallon to liter
liter_per_gal = 3.785

# cubic meter to liter
liter_per_cu_m = 1000.0

# bar to Pa (1 bar = 1 * 10^5 Pa)
Pa_per_bar = 1.0 * 10**5

# psi to Pa (1 psi = 6894.76 Pa)
Pa_per_psi = 6894.76

# atm to Pa (1 atm = 101325 Pa)
Pa_per_atm = 101325.0

# kg to g (1 kg = 1000 g)
g_per_kg = 1000.0

# tonne to kg (1 tonne = 1000 kg)
kg_per_tonne = 1000.0

# tonne to kt (1 kt = 1000 tonne)
tonne_per_kt = 1000.0

# kmol to mol (1 kmol = 1000 mol)
mol_per_kmol = 1000.0

# kJ to J (1 kJ = 1000 J)
J_per_kJ = 1000.0

# days per year
day_per_yr = 365.0

# hours per day
hr_per_day = 24.0

# second per hour
sec_per_hr = 3600.0

# ----------------------------------------------------------------------------
# dollar year indices

# read in cost indices for dollar year conversion
df_cost_idx = pd.read_csv(this_file/pathlib.Path('inputs/cost_indices_2001_2022.csv'))

# create dataframe of CPI-U by year
cpi_u = df_cost_idx[['Year', 'CPI-U']].copy()
cpi_u.rename(columns = {'CPI-U': 'Cost Index'}, inplace = True)

# create dataframe of CEPCI by year
cepci = df_cost_idx[['Year', 'CEPCI']].copy()
cepci.rename(columns = {'CEPCI': 'Cost Index'}, inplace = True)
    
# ----------------------------------------------------------------------------
# CO2 transport cost

# read in all-in liquid CO2 trucking cost
df_co2 = pd.read_csv(
    this_file/pathlib.Path('inputs/liq_co2_trucking_costs.csv'), 
    usecols = [
        'Output Dollar Year (User Input)', 
        'Size (kt-CO2/y)', 
        'Distance (mi)', 
        'Total ($/t-CO2 gross)'
        ],
    dtype = 'float'
    )

#%% FUNCTIONS: GENERAL

# ----------------------------------------------------------------------------
# function: gas volume at normal conditions

def mol_to_norm_cu_m(
        num_mols
        ):
    """Calculate gas volume at normal conditions (Nm^3) for given number of moles. Works for molar flowrate to volumetric flowrate conversion if time unit is the same (e.g., mol/hr --> Nm^3/hr).
    
    Parameters
    ----------
    num_mols : float
        Number of moles of gas.
    
    Returns
    -------
    gas_vol_norm_cu_m
        Gas volume (Nm^3) (or volumetric flowrate, Nm^3/time).
    """
    gas_vol_norm_cu_m = \
        num_mols * gas_const_kJ_per_kmol_K * \
        norm_temp_K / norm_pres_Pa
    
    return gas_vol_norm_cu_m

# ----------------------------------------------------------------------------
# function: dollar year conversion

def dollar_year_conversion(
        output_dollar_year, 
        cost_index = cepci, 
        input_dollar_year = None, 
        input_cost_index = None
        ):
    """Calculate dollar year conversion (multiplier) using user-selected cost index. Either input dollar year (e.g., 2006) or input cost index (e.g., CEPCI = 1000) needs to be provided.
    
    Parameters
    ----------
    output_dollar_year : int
        Target (output) dollar year.
    cost_index : dataframe, optional
        Cost index (CEPCI or CPI-U) by year. Default: CEPCI.
    input_dollar_year : int, optional
        Input data dollar year.
    input_cost_index : float, optional
        User-defined input cost index, e.g., CEPCI = 1000.
    
    Returns
    -------
    dollar_year_multiplier
        Dollar year multiplier to convert from input dollar year to target (output) dollar year.
    """    
    # raise error if neither input dollar year nor input cost index is provided
    if (input_dollar_year == None) and (input_cost_index == None):
        raise ValueError(
            'Either input dollar year or input cost index needs to be provided.'
            )
    
    # priorize using input dollar year (if provided) to calculate 
    # input cost index
    if input_dollar_year != None:
        input_cost_index = \
            cost_index['Cost Index'].loc[
                cost_index['Year'] == input_dollar_year
                ].values[0]

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year        
    dollar_year_multiplier = \
        cost_index['Cost Index'].loc[
            cost_index['Year'] == output_dollar_year
            ].values[0] / input_cost_index

    return dollar_year_multiplier

# ----------------------------------------------------------------------------
# function: net present value (NPV) using MACRS depreciation

def MACRS_depreciation_NPV(
        depr_yr, 
        discount_rate = discount_rate
        ):
    """Calculate net present value (NPV) in % using MACRS depreciation.
    
    Parameters
    ----------
    depr_yr : int
        The MACRS depreciation schedule length (yr) of the target equipment.
    discount_rate : float, optional
        Discount rate. Default: real after-tax discount rate on "Financial Inputs" tab in HDSAM V3.1.
    
    Returns
    -------
    depr_NPV
        Net present value (NPV) in % using MACRS depreciation for the target equipment.
    """    
    # convert the type of input depreciation year from float to int
    if type(depr_yr) is float:
        depr_yr = int(depr_yr)
    
    # raise error if the input depreciation year is not in the MACRS 
    # depreciation table
    if str(depr_yr) not in df_macrs_depr_idx.columns.values:
        raise ValueError(
            'The input depreciation year is not in MACRS depreciation table'
            )
    
    # extract the column of the depreciation schedule corresponding to 
    # the input depreciation year
    depr_schedule = df_macrs_depr_idx[str(depr_yr)].dropna()
    
    # calculate the net present value (NPV) of the depreciation schedule
    # start at year 1 (the first depreciation needs to be discounted in 
    # calculation) according to HDSAM V3.1
    depr_NPV = 0
    for i, depreciation in enumerate(depr_schedule):
        depr_NPV += depreciation / (1 + discount_rate)**(i + 1)

    return depr_NPV

# ----------------------------------------------------------------------------
# function: capital cost levelization

def levelized_capital_cost(
        tot_cap_inv_usd, 
        life_yr, 
        depr_yr,
        input_dollar_year, 
        discount_rate = discount_rate, 
        total_tax_rate = total_tax_rate,
        output_dollar_year = None
        ):
    """Calculate levelized capital cost ($/yr, specified output dollar year).
    
    Parameters
    ----------
    tot_cap_inv_usd : float
        Total capital investment ($).
    life_yr: float
        Equipment lifetime (yr).
    depr_yr : int
        The MACRS depreciation schedule length (yr) of the target equipment.
    input_dollar_year : int
        Input data dollar year.
    discount_rate : float, optional
        Discount rate. Default: real after-tax discount rate on "Financial Inputs" tab in HDSAM V3.1.
    total_tax_rate : float, optional
        Total tax rate. Default: total tax rate including state taxes and federal taxes on "Financial Inputs" tab in HDSAM V3.1.
    output_dollar_year : int, optional
        Dollar year of calculated costs. Default: same as input dollar year.

    Returns
    -------
    lev_cap_cost_usd_per_yr
        Levelized capital cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # calculate capital recovery factor
    capital_recovery_factor = \
        discount_rate * (1 + discount_rate)**life_yr / \
        ((1 + discount_rate)**life_yr - 1)
    
    # calculate net present value (NPV) for given equipment
    # MACRS depreciation schedule length (yr)
    depr_NPV = MACRS_depreciation_NPV(depr_yr, discount_rate)
        
    # calculate real present value of depreciation 
    real_PV_depr = depr_NPV * (1 + discount_rate)
        
    # calculate real fixed charge rate
    real_fixed_charge_rate = \
        capital_recovery_factor * (1 - real_PV_depr * total_tax_rate) / \
        (1 - total_tax_rate)

    # assume output dollar year is same as input dollar year by default
    if output_dollar_year == None:
        output_dollar_year = input_dollar_year

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate levelized capital cost ($/yr, specified output dollar year)
    lev_cap_cost_usd_per_yr = \
        tot_cap_inv_usd * \
        real_fixed_charge_rate * dollar_year_multiplier
    
    return lev_cap_cost_usd_per_yr, \
        output_dollar_year
        
#%% FUNCTIONS: COMPRESSOR

# ----------------------------------------------------------------------------
# function: compressor power and sizing

# TODO: revisit default compressor efficiency - incorporate motor efficiency?
# TODO: revisit ideal gas assumption - compressibility as function of 
# pressure and temperature?
# TODO: revisit specific heat ratio

def compressor_power_and_size(
        out_pres_bar, 
        in_pres_bar, 
        in_temp_K, 
        gas_flow_mol_per_sec, 
        compr_eff = 0.75,
        compressibility = 1.0, 
        spec_heat_ratio = 1.41
        ):
    """Calculate compressor power (kW), size (kW/pump), number of compressor stages. Outputs validated using HDSAM V3.1.
    
    Parameters
    ----------
    out_pres_bar : float
        Compressor outlet pressure (bar).
    in_pres_bar : float  
        Compressor inlet pressure (bar).
    in_temp_K : float
        Compressor inlet temperature (bar).
    gas_flow_mol_per_sec : float
        Gas molar flowrate through compressor (mol/s).
    compr_eff : float, optional
        Compressor efficiency. Default: 0.75 (HDSAM V3.1).
    compressibility : float, optional
        Compressibility of gas through compressor. Default: 1 (ideal gas).
    spec_heat_ratio : float, optional
        Specific heat ratio (Cp/Cv) of gas through compressor. Default: 1.41 (hydrogen @ 20 deg.C).

    Returns
    -------
    compr_tot_power_kW
        Total compressor power (kW) of all stages. For energy calculations.
    compr_power_kW_per_stg
        Compressor power (kW) per stage. For fixed cost calculations.
    num_stgs
        Number of compressor stages required. For fixed cost calculations.
    """
    # define maximum pressure ratio per stage (source: HDSAM V3.1)
    max_pres_ratio_per_stg = 2.1

    # calculate number of compressor stages required
    pres_ratio = out_pres_bar / in_pres_bar
    num_stgs = math.ceil(
        math.log(pres_ratio) / \
        math.log(max_pres_ratio_per_stg)
        )
    
    # calculate compressor power per stage (kW)
    compr_power_kW_per_stg = \
        compressibility * \
        gas_flow_mol_per_sec / mol_per_kmol * \
        gas_const_kJ_per_kmol_K * in_temp_K * (1 / compr_eff) * \
        (spec_heat_ratio / (spec_heat_ratio - 1)) * \
        (pres_ratio**((spec_heat_ratio - 1) / \
                      spec_heat_ratio / num_stgs) - 1)
    
    # calculate total compressor power across all stages (kW)
    compr_tot_power_kW = compr_power_kW_per_stg * num_stgs
    
    return compr_tot_power_kW, \
        compr_power_kW_per_stg, \
        num_stgs

# ----------------------------------------------------------------------------
# function: compressor installed and O&M costs

def compressor_fixed_costs(
        compr_power_kW_per_stg, 
        num_stgs, 
        output_dollar_year
        ):
    """Calculate compressor installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    For now, use equation for "Refueling Station Main Compressors" for 700 bar refueling (HDSAM V3.1 "Cost Data" tab).
    
    Parameters
    ----------
    compr_power_kW_per_stg : float
        Compressor power (kW) per stage. 
    num_stgs : int
        Number of compressor stages.
    output_dollar_year : int
        Dollar year of calculated costs.
    
    Returns
    -------
    compr_inst_cost_usd
        Compressor installed cost ($, user-specified output dollar year). 
    compr_om_cost_usd_per_yr
        Compressor annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year
    # (dollar year in HDSAM V3.1 for refueling station main compressors)
    input_dollar_year = 2007

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate compressor uninstalled cost ($, output dollar year) 
    compr_uninst_cost_usd = \
        num_stgs * 40035.0 * compr_power_kW_per_stg**0.6038 * \
        dollar_year_multiplier

    # calculate compressor installed cost ($, output dollar year)
    compr_inst_cost_usd = compr_uninst_cost_usd * inst_factor
    
    # calculate compressor annual O&M cost ($/yr, output dollar year)
    # HDSAM V3.1: 4% of compressor installed cost
    compr_om_cost_usd_per_yr = 0.04 * compr_inst_cost_usd
    
    return compr_inst_cost_usd, \
        compr_om_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: PUMP

# ----------------------------------------------------------------------------
# function: pump power and sizing

# TODO: revisit maximum pump flowrate (capacity)
# TODO: incorporate motor efficiency?

def pump_power_and_size(
        out_pres_bar,
        in_pres_bar,
        fluid_flow_kg_per_sec, 
        dens_kg_per_cu_m = dens_liq_H2_kg_per_cu_m, 
        pump_eff = 0.75, 
        max_pump_flow_kg_per_hr = 120
        ):
    """Calculate pump power (kW), size (kW/pump), number of pumps. Outputs validated using HDSAM V3.1.
    
    Parameters
    ----------
    out_pres_bar : float
        Pump outlet pressure (bar).
    in_pres_bar : float  
        Pump inlet pressure (bar).
    fluid_flow_kg_per_sec : float
        Fluid flowrate through pump (kg/s).
    dens_kg_per_cu_m : float, optional
        Density of fluid through pump (kg/m^3). Default: liquid hydrogen.
    pump_eff : float, optional
        Pump efficiency. Default: 0.75 (HDSAM V3.1).
    max_pump_flow_kg_per_hr : float, optional
        Pump capacity (maximum flowrate) (kg/hr). Default: 120 kg/hr (HDSAM V3.1).

    Returns
    -------
    pump_tot_power_kW
        Total pump power (kW) of all stages. For energy calculations.
    pump_power_kW_per_pump
        Pump power (kW) per stage. For fixed cost calculations.
    num_pumps
        Number of pump stages required. For fixed cost calculations.
    """
    # calculate pressure difference between outlet and inlet streams (Pa)
    pres_diff_Pa = (
        out_pres_bar - in_pres_bar
        ) * Pa_per_bar

    # calculate total pump power (kW)
    pump_tot_power_kW = \
        fluid_flow_kg_per_sec * pres_diff_Pa / \
        dens_kg_per_cu_m / pump_eff / J_per_kJ

    # calculate number of pumps required and pump power per pump (kW)
    # no pump required if total pump power is zero
    if pump_tot_power_kW == 0.0:
        num_pumps = 0
        pump_power_kW_per_pump = 0.0
    else:
        num_pumps = math.ceil(
            fluid_flow_kg_per_sec * sec_per_hr / max_pump_flow_kg_per_hr
            )
        pump_power_kW_per_pump = pump_tot_power_kW / num_pumps
    
    return pump_tot_power_kW, \
        pump_power_kW_per_pump, \
        num_pumps

# ----------------------------------------------------------------------------
# function: high-pressure cryogenic pump installed and O&M costs

def cryo_pump_fixed_costs(
        num_pumps, 
        output_dollar_year
        ):
    """Calculate high-pressure cryogenic pump installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    For now, use equation for station pump for 700 bar dispensing (HDSAM V3.1 "Cost Data" tab).
    
    Parameters
    ----------
    num_pumps : int
        Number of pumps.
    output_dollar_year : int
        Dollar year of calculated costs.
    
    Returns
    -------
    pump_inst_cost_usd
        Pump installed cost ($, user-specified output dollar year). 
    pump_om_cost_usd_per_yr
        Pump annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for liquid hydrogen pumps at station)
    input_dollar_year = 2010

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate high-pressure cryogenic pump uninstalled cost 
    # ($, output dollar year) 
    pump_uninst_cost_usd = \
        num_pumps**0.8 * 700000.0 * dollar_year_multiplier
    
    # calculate high-pressure cryogenic pump installed cost 
    # ($, output dollar year)
    pump_inst_cost_usd = pump_uninst_cost_usd * inst_factor
    
    # calculate high-pressure cryogenic pump annual O&M cost 
    # ($/yr, output dollar year)
    # HDSAM V3.1: 4% of high-pressure cryogenic pump installed cost
    pump_om_cost_usd_per_yr = 0.04 * pump_inst_cost_usd
    
    return pump_inst_cost_usd, \
        pump_om_cost_usd_per_yr, \
        output_dollar_year

# ----------------------------------------------------------------------------
# function: low-head pump installed and O&M costs

def low_head_pump_fixed_costs(
        num_pumps, 
        fluid_flow_cu_m_per_hr, 
        output_dollar_year
        ):
    """Calculate low-head pump installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year. Outputs validated using HDSAM V3.1.
    
    "Low-Head LH2 Pump" cost equation on HDSAM V3.1 "Liquid H2 Terminal" tab is a function of hydrogen mass flowrate. Here, converted to function of volumetric flowrate using liquid hydrogen density (hence the different base cost, or multiplier, in the cost equation). Assume generally applicable to other fluids (e.g., formic acid).
    
    Parameters
    ----------
    num_pumps : int
        Number of pumps.
    fluid_flow_cu_m_per_hr : float
        Fluid flowrate through pump (m^3/hr).
    output_dollar_year : int
        Dollar year of calculated costs.
    
    Returns
    -------
    pump_inst_cost_usd
        Pump installed cost ($, user-specified output dollar year). 
    pump_om_cost_usd_per_yr
        Pump annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (inferred from "Low-Head LH2 Pump" cost on HDSAM V3.1 
    # "Liquid H2 Terminal" tab)
    input_dollar_year = 2016

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate low-head pump uninstalled cost ($, output dollar year) 
    if num_pumps == 0:
        pump_uninst_cost_usd = 0.0
    else:
        pump_uninst_cost_usd = \
            num_pumps * 19803.2 * (
                fluid_flow_cu_m_per_hr / num_pumps
                )**0.3431 * dollar_year_multiplier
    
    # calculate low-head pump installed cost ($, output dollar year)
    pump_inst_cost_usd = pump_uninst_cost_usd * inst_factor
    
    # calculate low-head pump annual O&M cost ($/yr, output dollar year)
    # HDSAM V3.1: 1% of low-head pump installed cost
    pump_om_cost_usd_per_yr = 0.01 * pump_inst_cost_usd
    
    return pump_inst_cost_usd, \
        pump_om_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: VAPORIZER (EVAPORATOR)

# NOTE: vaporizer energy consumption and energy cost are not considered in 
# HDSAM V3.1 due to their wide variation and anticipated low impact on 
# overall costs.

# ----------------------------------------------------------------------------
# function: vaporizer installed and O&M costs

# TODO: use vaporizer capacity in Nexant et al., "Interim Report" to 
# calculate number of vaporizers needed; number of vaporizers in HDSAM V3.1 
# appears to be hardcoded to be one

def vaporizer_fixed_costs(
        fluid_flow_kg_per_hr,
        output_dollar_year,
        vap_capacity_kg_per_hr = 250
        ):
    """Calculate vaporizer (evaporator) installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    For now, focus on vaporizer (evaporator) at refueling station. Evaporator cost equation available for terminal, but evaporator capacity at terminal is zero in HDSAM V3.1. 
        
    Parameters
    ----------
    fluid_flow_kg_per_hr : float
        Fluid flowrate through vaporizer (kg/s).    
    output_dollar_year : int
        Dollar year of calculated costs.    
    vap_capacity_kg_per_hr : float, optional
        Vaporizer capacity (maximum flowrate) (kg/hr). Default: 250 kg/hr (HDSAM V3.1).
    
    Returns
    -------
    vap_inst_cost_usd
        Vaporizer installed cost ($, user-specified output dollar year). 
    vap_om_cost_usd_per_yr
        Vaporizer annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for refueling station evaporizer)
    input_dollar_year = 2013

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    
    
    # calculate number of vaporizers needed
    num_vaps = fluid_flow_kg_per_hr / vap_capacity_kg_per_hr

    # calculate heat exchanger uninstalled cost ($, output dollar year) 
    vap_uninst_cost_usd = \
        num_vaps * (
            1000.0 * vap_capacity_kg_per_hr + 15000.0
            ) * dollar_year_multiplier

    # calculate vaporizer installed cost ($, output dollar year)
    vap_inst_cost_usd = vap_uninst_cost_usd * inst_factor

    # calculate vaporizer annual O&M cost ($/yr, output dollar year)
    # HDSAM V3.1: 1% of vaporizer installed cost
    vap_om_cost_usd_per_yr = 0.01 * vap_inst_cost_usd
       
    return vap_inst_cost_usd, \
        vap_om_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: HEAT EXCHANGER
       
# ---------------------------------------------------------------------------- 
# function: heat exchanger energy

# TODO: revisit heat exchanger sizing (capacity, number of equipment)
# TODO: add "overhead precooling" energy for compressed hydrogen? 
# (overhead precooling ~50% refrigeration energy at station)
# (overhead precooling ~15% onsite energy consumption at station)     

def heat_exchanger_energy(
        out_temp_K,
        in_temp_K, 
        molar_mass_kg_per_kmol = molar_mass_H2_kg_per_kmol, 
        hx_eff = 0.9, 
        spec_heat_ratio = 1.41
        ):
    """Calculate heat exchanger energy (kWh/kg fluid). Outputs validated using HDSAM V3.1.
    
    Parameters
    ----------
    out_temp_K : float
        Heat exchanger outlet temperature (K).        
    in_temp_K : float  
        Heat exchanger inlet temperature (K).        
    molar_mass_kg_per_kmol : float, optional
        Molar mass of fluid through heat exchanger (kg/kmol, or g/mol). Default: hydrogen.        
    hx_eff : float, optional
        Heat exchanger efficiency. Default: 0.9 (HDSAM V3.1 refrigerator COP).
    spec_heat_ratio : float, optional
        Specific heat ratio (Cp/Cv) of gas through compressor. Default: 1.41 (hydrogen @ 20 deg.C).

    Returns
    -------
    hx_elec_kWh_per_kg
        Heat exchanger energy (kWh/kg fluid)
    """
    # calculate heat exchanger energy (kWh/kg fluid)
    hx_elec_kWh_per_kg = abs(
        spec_heat_ratio / (spec_heat_ratio - 1) * \
        gas_const_kJ_per_kmol_K / molar_mass_kg_per_kmol * \
        (out_temp_K - in_temp_K) / hx_eff / sec_per_hr
        )

    return hx_elec_kWh_per_kg  

# ----------------------------------------------------------------------------
# function: heat exchanger installed and O&M costs

def heat_exchanger_fixed_costs(
        out_temp_K, 
        num_hx, 
        output_dollar_year,
        hx_capacity_ton_per_unit = 3.4
        ):
    """Calculate heat exchanger installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    Uninstalled cost: use purchase cost equation for "Hydrogen Precooling Refrigeration Equipment" (HDSAM V3.1 "Cost Data" tab). 
    
    Installation cost factor: use hardcoded number (= 2) from "Refueling Station - Gaseous H2" tab in HDSAM V3.1, *not* installation cost adjustment equations on "Cost Data" tab. 
    
    Above equations/values are used for refueling station refrigerator cost calculations.
            
    Parameters
    ----------
    out_temp_K : float
        Heat exchanger outlet temperature (K).        
    num_hx : int
        Number of heat exchangers.
    output_dollar_year : int
        Dollar year of calculated costs.    
    hx_capacity_ton_per_unit : float, optional
        Vaporizer capacity (ton/unit). Default: 3.4 ton/unit.
        
    Returns
    -------
    hx_inst_cost_usd
        Heat exchanger installed cost ($, user-specified output dollar year). 
    hx_om_cost_usd_per_yr
        Heat exchanger annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for hydrogen precooling refrigeration 
    # equipment)
    input_dollar_year = 2013

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 2

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate heat exchanger uninstalled cost ($, output dollar year) 
    hx_uninst_cost_usd = \
        1.25 * 11092.0 * (
            100.0 * num_hx * hx_capacity_ton_per_unit / out_temp_K
            )**0.8579 * dollar_year_multiplier

    # calculate heat exchanger installed cost ($, output dollar year)
    hx_inst_cost_usd = hx_uninst_cost_usd * inst_factor
    
    # calculate heat exchanger annual O&M cost ($/yr, output dollar year)
    # HDSAM V3.1: 2% of heat exchanger installed cost
    hx_om_cost_usd_per_yr = 0.02 * hx_inst_cost_usd
    
    return hx_inst_cost_usd, \
        hx_om_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: LIQUEFIER

# ----------------------------------------------------------------------------
# function: liquefier energy

def liquefier_energy(
        liquef_size_tonne_per_day
        ):
    """Calculate liquefier energy (kWh/kg H2). Varies with assumed liquefier size (tonne H2/day), using relationship in HDSAM V3.1. Outputs validated using HDSAM V3.1.
    
    Parameters
    ----------
    liquef_size_tonne_per_day : float
        Liquefier size (capacity) (tonne H2/day).
    
    Returns
    -------
    liquef_elec_kWh_per_kg
        Liquefier energy (kWh/kg H2)
    """
    # calculate liquefier energy (kWh/kg H2)
    liquef_elec_kWh_per_kg = \
        13.382 * liquef_size_tonne_per_day**(-0.1)
    
    return liquef_elec_kWh_per_kg

# ----------------------------------------------------------------------------
# function: liquefier installed and O&M costs

# TODO: calculate number of liquefiers needed based on 
# target hydrogen station capacity and size of liquefier

def liquefier_fixed_costs(
        liquef_size_tonne_per_day,
        output_dollar_year
        ):
    """Calculate liquefier installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year. Installed cost scales with assumed liquefier size (tonne H2/day), using relationship in HDSAM V3.1. Outputs validated using HDSAM V3.1.
            
    Parameters
    ----------
    liquef_size_tonne_per_day : float
        Liquefier size (capacity) (tonne H2/day).
    output_dollar_year : int
        Dollar year of calculated costs.    
        
    Returns
    -------
    liquef_inst_cost_usd
        Liquefier installed cost ($, user-specified output dollar year). 
    liquef_om_cost_usd_per_yr
        Liquefier annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for liquefier costs)
    input_dollar_year = 2014
    
    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate liquefier installed cost ($, output dollar year) 
    liquef_inst_cost_usd = \
        5.6e6 * liquef_size_tonne_per_day**0.8 * dollar_year_multiplier
    
    # calculate liquefier annual O&M cost ($/yr, output dollar year)
    # HDSAM V3.1: 1% of liquefier installed cost
    liquef_om_cost_usd_per_yr = 0.01 * liquef_inst_cost_usd
    
    return liquef_inst_cost_usd, \
        liquef_om_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: REACTOR

# ----------------------------------------------------------------------------
# function: reactor installed and O&M costs

# TODO: revisit installed cost factor and O&M cost as % of installed cost 
# (placeholders for now)
# TODO: revisit reactor volume limits - coordinate with Components team

def reactor_fixed_costs(
        react_vol_cu_m, 
        react_pres_bar, 
        num_reacts, 
        output_dollar_year, 
        method = 'woods'
        ):
    """Calculate reactor installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    Types:
    (a) Jacketed and stirred reactor, carbon steel. (Peters, Timmerhaus, West, Figure 13.15.)
    (b) Stirred tank reactor: batch (backmix). Pressure vessel, jacketed with agitation, 0.3 MPa. Vertical cylinder, dished ends, jacketed agitated vessel, jacket rated at 0.8 MPa, 175 C, top entry agitator with packed seal, 1–2 kW/m^3, 9 nozzles, 4 baffles. FOB c/s [free-on-board, carbon steel] cost including jacket, top entry agitator and motor. (Woods, Section 6.27.)
        
    Reference(s): 
    (a) Peters, Timmerhaus, West, 2003, "Plant Design and Economics for Chemical Engineers"
    (b) Woods, 2007, "Rules of Thumb Engineering Practice"

    Parameters
    ----------
    react_vol_cu_m : float
        Reactor volume (m^3). 
    react_pres_bar : float
        Reaction pressure (bar).
    num_reacts : int
        Number of reactors.
    output_dollar_year : int
        Dollar year of calculated costs. 
    method : str
        Method (reference) used for reactor cost estimates. Currently accepted methods: ['woods', 'peters et al.']. Default: 'woods'.
        
    Returns
    -------
    react_inst_cost_usd
        Reactor installed cost ($, user-specified output dollar year). 
    react_om_cost_usd_per_yr
        Reactor annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    ##########################################################################
    # check user-supplied method
    ##########################################################################
    
    # accepted methods
    methods = [
        'woods', 
        'peters et al.' 
        ]
    
    # output error message if user-supplied method is not one of 
    # the accepted methods
    if method not in methods:
        raise ValueError(
            'Invalid method. Accepted methods: {}'.format(methods)
        )

    ##########################################################################
    # inputs and assumptions for all methods
    ##########################################################################

    # cost multiplier for pressure rating
    # source: Woods, 2007. 
    # NOTE: Calculated pressure factors using correlations in 
    # Peters, Timmerhaus, West, 2003 are similar for the 
    # pressure ratings available (3.45 bar, 20.7 bar, 103.5 bar).
    # NOTE: Reactor costs are available for pressure up to 350 bar in Woods;
    # costs above 350 bar are extrapolated.
    if react_pres_bar <= 3.0:
        pres_factor = 1.0
    elif react_pres_bar <= 21.0:
        pres_factor = 1.25
    elif react_pres_bar <= 40.0:
        pres_factor = 2.4
    elif react_pres_bar <= 100.0:
        pres_factor = 8.2
    elif react_pres_bar <= 200.0:
        pres_factor = 11.0
    elif react_pres_bar <= 350.0:
        pres_factor = 12.0
    else:
        print (
            'Reactor costs for pressure > 350 bar '
            'are extrapolated.'
            )
        pres_factor = 0.3519 * react_pres_bar**0.6083
    
    ##########################################################################
    # reactor cost following Woods
    ##########################################################################
    
    if method == 'woods':
        
        #--------------------------------------------------------------------#
        # inputs and assumptions
        #--------------------------------------------------------------------#

        # input CEPCI
        # CEPCI associated with costs in Woods, 2007
        input_cepci = 1000
        
        # installation factor
        # Woods, 2007: "L+M*" (labor and materials) factor;
        # converts free-on-board (FOB) cost to bare module cost. 
        # "*" denotes the exlusion of instrumentation material and 
        # labor costs.
        # L+M* for stirred tank reactor (pressure vessel, 
        # jacketed with agitation): 2.25–2.52; 
        # high value for installation of one unit, low value for many units.
        inst_factor = 2.5
        
        # reference, minimum, and maximum reactor volume (m^3)
        # reference = base size for cost scaling.
        # minimum and maximum = size for which cost scaling
        # is available in Woods, 2007.
        ref_react_vol_cu_m = 3.0
        min_react_vol_cu_m = 0.3
        max_react_vol_cu_m = 90.0
        
        # specify reference FOB cost ($, input CEPCI)
        ref_react_purc_cost_usd = 75000.0

        #--------------------------------------------------------------------#
        # calculations
        #--------------------------------------------------------------------#

        # calculate conversion (multiplier) from input dollar year to 
        # output dollar year
        dollar_year_multiplier = \
            dollar_year_conversion(
                output_dollar_year = output_dollar_year, 
                input_cost_index = input_cepci
                )
        
        # override reactor volume if user input is below minimum
        # print message if user input is above maximum
        if react_vol_cu_m < min_react_vol_cu_m:
            react_vol_cu_m = min_react_vol_cu_m
        elif react_vol_cu_m > max_react_vol_cu_m:
            print (
                'Reactor volume is larger than maximum size (90 m^3) '
                'for which cost data is available. Consider resizing.'
                )
        
        # determine scaling factor (exponent) based on reactor volume
        # NOTE: Scaling factor = 0.53 for 3-90 m^3 in Woods. Relax upperbound 
        # here because reactor volume and number of reactors are upstream 
        # model outputs. Otherwise could resize here for cost calculations.
        if (react_vol_cu_m >= min_react_vol_cu_m) and \
            (react_vol_cu_m < ref_react_vol_cu_m):
            scaling_factor = 0.4
        elif react_vol_cu_m >= ref_react_vol_cu_m:
            scaling_factor = 0.53

    ##########################################################################
    # reactor cost following Peters, Timmerhaus, West
    ##########################################################################
    
    if method == 'peters et al.':

        #--------------------------------------------------------------------#
        # inputs and assumptions
        #--------------------------------------------------------------------#

        # specify input dollar year
        input_dollar_year = 2002

        # specify installation factor 
        # (scales uninstalled cost to installed cost)
        # source: Peters, Timmerhaus, West, Table 6-5
        # range for metal tanks: 1.3-1.6
        inst_factor = 1.5
        
        # specify reference reactor volume (m^3) and
        # reference purchase cost ($, input dollar year)
        # http://www.mhhe.com/engcs/chemical/peters/data/ce.html
        # purchased cost of jacketed and stirred reactors, 
        # carbon steel, 345 kPa
        ref_react_vol_cu_m = 1.0
        ref_react_purc_cost_usd = 11984.0
        
        # specify scaling factor (exponent)
        # calculated from curve fitting
        scaling_factor = 0.546

        #--------------------------------------------------------------------#
        # calculations
        #--------------------------------------------------------------------#

        # calculate conversion (multiplier) from input dollar year to 
        # output dollar year
        dollar_year_multiplier = \
            dollar_year_conversion(
                input_dollar_year = input_dollar_year, 
                output_dollar_year = output_dollar_year
                )    

    ##########################################################################
    # reactor installed and fixed O&M cost
    ##########################################################################
    
    # calculate reactor purchase or FOB cost ($, output dollar year) 
    react_purc_cost_usd = \
        num_reacts * ref_react_purc_cost_usd * \
        (react_vol_cu_m / ref_react_vol_cu_m)**scaling_factor * \
        pres_factor * dollar_year_multiplier
        
    # calculate reactor installed cost ($, output dollar year)
    react_inst_cost_usd = react_purc_cost_usd * inst_factor

    # calculate reactor annual O&M cost ($/yr, output dollar year)
    react_om_cost_usd_per_yr = 0.01 * react_inst_cost_usd

    return react_inst_cost_usd, \
        react_om_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: ELECTROLYZER

# ----------------------------------------------------------------------------
# function: electrolyzer energy

# TODO: add electrolyzer energy as function


# ----------------------------------------------------------------------------
# function: electrolyzer installed and O&M costs

# TODO: revisit installation factor
# TODO: revisit installed cost factor and O&M cost as % of installed cost 

def electrolyzer_fixed_costs(
        electr_area_sq_m, 
        output_dollar_year, 
        electr_purc_cost_usd_per_sq_m = 30000.0,
        electr_cost_dollar_year = 2019
        ):
    """Calculate electrolyzer installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
            
    Reference(s): 
    (a) Ramdin, M., Morrison, A. R. T., de Groen, M., van Haperen, R., de Kler, R., van den Broeke, L. J. P., Trusler, J. P. M., de Jong, W., & Vlugt, T. J. H. (2019). High Pressure Electrochemical Reduction of CO2 to Formic Acid/Formate: A Comparison between Bipolar Membranes and Cation Exchange Membranes. Industrial & Engineering Chemistry Research, 58(5), 1834-1847. https://doi.org/10.1021/acs.iecr.8b04944.
    (b) Li, W. Q., J. T. Feaster, S. A. Akhade, J. T. Davis, A. A. Wong, V. A. Beck, . . . S. E. Baker. 2021. "Comparative Techno-Economic and Life Cycle Analysis of Water Oxidation and Hydrogen Oxidation at the Anode in a CO2 Electrolysis to Ethylene System." ACS Sustainable Chemistry & Engineering 9 (44): 14678-14689. https://doi.org/10.1021/acssuschemeng.1c01846.


    Parameters
    ----------
    electr_area_sq_m : float
        Electrolyzer area (m^2). 
    output_dollar_year : int
        Dollar year of calculated costs. 
    electr_purc_cost_usd_per_sq_m : float, optional
        Electrolyzer purchase cost ($/m^2). Default: CO2 electrolyzer in Ramdin et al., 2019.
    electr_cost_dollar_year : int, optional
        Input dollar year of electrolyzer cost. Default: 2019 (publication year of Ramdin et al., 2019).
        
    Returns
    -------
    electr_inst_cost_usd
        Electrolyzer installed cost ($, user-specified output dollar year). 
    electr_om_cost_usd_per_yr
        Electrolyzer annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3
        
    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = electr_cost_dollar_year, 
            output_dollar_year = output_dollar_year
            )    
    
    # calculate electrolyzer purchase cost ($, output dollar year) 
    electr_purc_cost_usd = \
        electr_purc_cost_usd_per_sq_m * \
        electr_area_sq_m * dollar_year_multiplier
        
    # calculate electrolyzer installed cost ($, output dollar year)
    electr_inst_cost_usd = electr_purc_cost_usd * inst_factor

    # calculate electrolyzer annual O&M cost ($/yr, output dollar year)
    electr_om_cost_usd_per_yr = 0.01 * electr_inst_cost_usd

    return electr_inst_cost_usd, \
        electr_om_cost_usd_per_yr, \
        output_dollar_year
        
#%% FUNCTIONS: SEPARATOR (PRESSURE SWING ADSORPTION, PSA)

# ----------------------------------------------------------------------------
# function: separator (PSA) power

def psa_power(
        in_flow_norm_cu_m_per_hr
        ):
    """Calculate separator (pressure swing adsorption, PSA) power (kW).
    
    Applies to electricity only, scales with inlet gas volume (Nm^3), based on Jouny et al., 2020 (citing Paturska et al., 2015, citing Bauer et al., 2013). 
    
    Note: energy scaler (multiplier) = 0.25 kWh/m^3 in Jouny et al., 2020; 0.23 kWh/m^3 in Paturska et al., 2015. Use Jouny et al., 2020 to be consistent with cost calculations.
    
    Parameters
    ----------
    in_flow_norm_cu_m_per_hr : float
        Separator inlet flowrate (Nm^3/hr).
    
    Returns
    -------
    psa_power_kW
        Separator (PSA) power (kW).
    """
    # calculate pressure swing adsorption power (kW)
    psa_power_kW = 0.25 * in_flow_norm_cu_m_per_hr
    
    return psa_power_kW

# ----------------------------------------------------------------------------
# function: separator (PSA) installed and O&M costs

# TODO: revisit assumption that cost calculated from Jouny et al., 2020 
# represents *installed* cost
# TODO: revisit O&M cost as % of installed cost

def psa_fixed_costs(
        in_flow_norm_cu_m_per_hr, 
        output_dollar_year
        ):
    """Calculate separator (pressure swing adsorption, PSA) installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    Investment scales with inlet gas flowrate (Nm^3/hr), based on Jouny et al., 2020 (citing Paturska et al., 2015, citing Bauer et al., 2013). Outputs validated using Paturska et al., 2015, Table 2.

    Default assumption: input dollar year = 2020 (year of publication of Jouny et al.). 
    
    Note that currency year in original reference (Bauer et al., 2013) is not specified, and there seems to be some inconsistency in currency year and exchange rate convension (e.g., cost from Paturska et al. and Bauer et al. is converted from EUR to USD in Jouny et al. using ~2020 exchange rate). For simplicity and easier tracking, assume 2020 dollar year. Costs in Bauer et al. are based on limited data; inconsistency in dollar year likely within cost uncertainty range.

    Parameters
    ----------
    in_flow_norm_cu_m_per_hr : float
        Separator inlet flowrate (Nm^3/hr).
    output_dollar_year : int
        Dollar year of calculated costs.    
        
    Returns
    -------
    psa_inst_cost_usd
        Separator (PSA) installed cost ($, user-specified output dollar year). 
    psa_om_cost_usd_per_yr
        Separator (PSA) annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (see notes above on dollar year)
    input_dollar_year = 2020

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate pressure swing adsorber investment ($, output dollar year)
    # assume this cost represents installed cost
    psa_inst_cost_usd = \
        1990000.0 * (
            in_flow_norm_cu_m_per_hr / 1000.0)**(1 - 0.67) * \
            dollar_year_multiplier

    # calculate separator annual O&M cost ($/yr, output dollar year)
    psa_om_cost_usd_per_yr = 0.01 * psa_inst_cost_usd

    return psa_inst_cost_usd, \
        psa_om_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: TRANSPORT

# ----------------------------------------------------------------------------
# function: transport fuel consumption and delivery requirements

# TODO: incorporate losses
# TODO: incorporate annual truck availability

def transport_energy(
        deliv_dist_mi, 
        speed_mi_per_hr, 
        load_unload_time_hr,
        fuel_econ_mi_per_gal,
        deliv_capacity_kg, 
        cargo_flow_kg_per_day, 
        vehicle_use_hr_per_day = 18
        ): 
    """Calculate transport fuel consumption (gallon/kg cargo), number of vehicles required, number of deliveries per day, total trip time (hr/trip). Outputs validated using HDSAM V3.1.
    
    HDSAM V3.1 nomenclature: number of deliveries = number of vehicle-trips (not necessarily integer)

    Fuel consumption varies with delivery distance (mile/trip), travel speed (mile/hr), loading and unloading time (hr), delivered capacity (kg H2/vehicle-trip), hydrogen station capacity (kg H2/day). 
    
    Calculations should hold for any form of transport. 
            
    Parameters
    ----------
    deliv_dist_mi : float
        Delivery distance (mile).
    speed_mi_per_hr : float
        Vehicle speed (mile/hr).
    load_unload_time_hr : float
        Total loading and unloading time (hr).
    fuel_econ_mi_per_gal : float
        Vehile fuel economy (mile/gallon).
    deliv_capacity_kg : float
        Vehicle delivered capacity (kg/vehicle/trip). Note: delivered capacity is often slightly smaller than nominal capacity.
    cargo_flow_kg_per_day : float
        Required flowrate of cargo (kg/day).
    vehicle_use_hr_per_day : float, optional
        Vehicle daily availability (hr/day). Default: 18 hr/day (HDSAM V3.1).

    Returns
    -------
    transport_fuel_gal_per_kg
        Transport fuel consumption (gallon/kg cargo).
    num_vehicles
        Number of vehicles (integer). For vehicle capital cost calculations.
    num_delivs_per_day
        Number of deliveries per day (float). For O&M cost calculations.
    trip_time_hr
        Total trip time (hr/trip), including travel, loading, and unloading.
    """
    # calculate number of deliveries per day
    num_delivs_per_day = cargo_flow_kg_per_day / deliv_capacity_kg

    # calculate total trip time (hr/trip)
    # including travel, loading, and unloading
    trip_time_hr = \
        deliv_dist_mi / speed_mi_per_hr + \
        load_unload_time_hr
    
    # calculate number of trips per day (trip/day)
    # time basis = per day 
    # to be consistent with hydrogen station capacity
    num_trips_per_day = vehicle_use_hr_per_day / trip_time_hr

    # caculate number of vehicles required (integer)
    num_vehicles = math.ceil(num_delivs_per_day / num_trips_per_day)
    
    # calculate truck fuel consumption (gallon/kg cargo)
    # based on number of deliveries (not necessarily integer in HDSAM V3.1)
    # NOTE: fuel economy (mile/gallon) is implicitly on a per vehicle basis
    transport_fuel_gal_per_kg = \
        deliv_dist_mi / \
        truck_fuel_econ_mi_per_gal * \
        num_delivs_per_day / cargo_flow_kg_per_day

    return transport_fuel_gal_per_kg, \
        num_vehicles, \
        num_delivs_per_day, \
        trip_time_hr
       
# ----------------------------------------------------------------------------
# function: compressed gaseous hydrogen truck capital cost        

# TODO: revisit number of trailer bundles
# for now, assume equal to number of trucks, but number of trailer bundles 
# in HDSAM V3.1 are more than 2x number of trucks

def gas_truck_capital_cost(
        num_trucks,
        output_dollar_year
        ):
    """Calculate compressed gaseous hydrogen truck capital cost ($) in user-specified output dollar year.
    
    Includes truck cab ("tractor") and tube tank trailer costs. Use "540 bar GH2 trailer cost" (HDSAM V3.1 "Cost Data" tab).

    Parameters
    ----------
    num_trucks : int
        Number of trucks.
    output_dollar_year : int
        Dollar year of calculated costs.    
        
    Returns
    -------
    gas_truck_cap_cost_usd
        Compressed gaseous hydrogen truck capital cost ($, user-specified output dollar year). 
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for compressed gaseous hydrogen trucks)
    input_dollar_year = 2012
    
    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate compressed gaseous hydrogen truck capital cost 
    # ($, output dollar year) 
    gas_truck_cap_cost_usd = \
        num_trucks * (115000 + 1100000) * dollar_year_multiplier
    
    return gas_truck_cap_cost_usd, \
        output_dollar_year  

# ----------------------------------------------------------------------------
# function: liquid hydrogen truck capital cost

def liquid_truck_capital_cost(
        num_trucks,
        output_dollar_year
        ):
    """Calculate liquid hydrogen truck capital cost ($) in user-specified output dollar year.
    
    Includes truck cab ("tractor") and tube tank trailer costs. Use "LH2 trailer cost" (HDSAM V3.1 "Cost Data" tab).

    Use same calculations for LOHC (e.g., formic acid) trucks.

    Parameters
    ----------
    num_trucks : int
        Number of trucks.
    output_dollar_year : int
        Dollar year of calculated costs.    
        
    Returns
    -------
    liq_truck_cap_cost_usd
        Liquid hydrogen truck capital cost ($, user-specified output dollar year). 
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for liquid hydrogen trucks)
    input_dollar_year = 2014
    
    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate liquid hydrogen truck capital cost ($, output dollar year) 
    liq_truck_cap_cost_usd = \
        num_trucks * (115000 + 950000) * dollar_year_multiplier
    
    return liq_truck_cap_cost_usd, \
        output_dollar_year
        
# ----------------------------------------------------------------------------
# function: truck O&M cost

# TODO: revisit allocation of O&M cost to tractor vs. trailers

def truck_om_cost(
        deliv_dist_mi, 
        num_delivs_per_yr, 
        output_dollar_year
        ):
    """Calculate truck annual O&M cost ($/yr) in user-specified output dollar year. Outputs validated using HDSAM V3.1.
    
    Calculations apply to both compressed and liquid hydrogen trucks in HDSAM. Use same calculations for LOHC (e.g., formic acid) trucks.
    
    Truck O&M costs include insurance, licensing and permits, operating, maintenance and repairs. 
    
    Parameters
    ----------
    deliv_dist_mi : float
        Delivery distance (mile).
    num_delivs_per_yr : float
        Number of deliveries per year.
    output_dollar_year : int
        Dollar year of calculated costs.    
        
    Returns
    -------
    tot_truck_om_cost_usd_per_yr
        Truck annual O&M cost ($/yr, user-specified output dollar year). 
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for truck fixed O&M costs)
    input_dollar_year = 2016

    # specifiy O&M cost components per mile ($/mile, implied: per truck)
    # truck O&M costs include insurance, licensing and permits, operating,
    # maintenance and repairs
    # HDSAM V3.1: insurance cost = $0.051/mile, doubled for hazardous cargo
    # HDSAM V3.1: licensing and permits cost = $0.056/mile, doubled for 
    # hazardous cargo
    # HDSAM V3.1: operating, maintenance and repair cost = $0.056/mile for 
    # outside maintenance and repairs; plus $0.019/mile for tires
    truck_om_cost_usd_per_mi_insurance = 0.051 * 2
    truck_om_cost_usd_per_mi_licensing = 0.056 * 2
    truck_om_cost_usd_per_mi_maintenance = 0.056 + 0.019

    # calculate total O&M cost per mile ($/mile, implied: per truck)
    tot_truck_om_cost_usd_per_mi = \
        truck_om_cost_usd_per_mi_insurance + \
        truck_om_cost_usd_per_mi_licensing + \
        truck_om_cost_usd_per_mi_maintenance

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate truck annual total O&M cost ($/yr, output dollar year)
    tot_truck_om_cost_usd_per_yr = \
        tot_truck_om_cost_usd_per_mi * \
        deliv_dist_mi * \
        num_delivs_per_yr * dollar_year_multiplier
    
    return tot_truck_om_cost_usd_per_yr, \
        output_dollar_year
        
# ----------------------------------------------------------------------------
# function: truck labor cost

def truck_labor_cost(
        num_delivs_per_yr, 
        trip_time_hr, 
        output_dollar_year, 
        truck_labor_rate_usd_per_hr = 20.43, 
        input_dollar_year = 2014
        ):
    """Calculate truck annual labor cost ($/yr) in user-specified output dollar year. Outputs validated using HDSAM V3.1.
    
    Labor cost includes overhead and general and administrative (G&A). Calculations apply to both compressed and liquid hydrogen trucks in HDSAM V3.1. Use same calculations for LOHC (e.g., formic acid) trucks.
    
    Parameters
    ----------
    num_delivs_per_yr : float
        Number of deliveries per year.
    trip_time_hr
        Total trip time (hr/trip), including travel, loading, and unloading.
    output_dollar_year : int
        Dollar year of calculated costs.    
    truck_labor_rate_usd_per_hr : float, optional
        Truck hourly labor rate ($/hr). Default: data from HDSAM V3.1 "Cost Data" tab.
    input_dollar_year: int, optional
        Input data dollar year. Default: data from HDSAM V3.1 "Cost Data" tab.
        
    Returns
    -------
    truck_labor_cost_usd_per_yr
        Truck annual labor cost ($/yr, user-specified output dollar year). 
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify overhead and G&A percentage (% unburdened labor cost)
    # HDSAM V3.1: 20% of unburdened truck labor cost
    truck_labor_cost_overhead_perc = 0.2

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate annual truck labor hours (hr/yr)
    truck_labor_hr_per_yr = num_delivs_per_yr * trip_time_hr

    # calculate annual unburdened truck labor cost ($/yr, output dollar year)
    unburdened_truck_labor_cost_usd_per_yr = \
        truck_labor_hr_per_yr * truck_labor_rate_usd_per_hr * \
        dollar_year_multiplier

    # calculate annual truck labor cost, including overhead and G&A 
    # ($/yr, output dollar year)
    truck_labor_cost_usd_per_yr = \
        unburdened_truck_labor_cost_usd_per_yr * \
        (1 + truck_labor_cost_overhead_perc)
    
    return truck_labor_cost_usd_per_yr, \
        output_dollar_year

# ----------------------------------------------------------------------------
# function: CO2 all-in transport cost (includes conditioning)

# TODO: use scipy.interpolate (fix DLL load error)

def CO2_transport_all_in_cost(
        CO2_flow_kt_per_yr,
        deliv_dist_mi,
        output_dollar_year
        ):
    """Interpolate all-in liquid CO2 trucking cost ($) for given amount of CO2 (kt/yr) and transport distance (mile). Output is in user-specified output dollar year.
    
    All-in costs include preconditioning (e.g., liquefaction) and trucking fixed and variable costs. Reference: Multimodal CO2 Transportation Cost Model developed at Lawrence Livermore National Laboratory under the auspices of the U.S. Department of Energy under Contract DE-AC52-07NA27344. Model available at: https://github.com/myers79/MuMo-CoCo. 

    Parameters
    ----------
    CO2_flow_kt_per_yr : float
        Amount of CO2 to be transported (kt/yr).
    deliv_dist_mi : float
        Delivery distance (mile).
    output_dollar_year : int
        Dollar year of calculated costs.    
        
    Returns
    -------
    liq_CO2_trucking_cost_usd_per_tCO2
        All-in liquid CO2 trucking cost ($/tonne CO2, user-specified output dollar year). 
    liq_CO2_trucking_cost_usd_per_yr
        All-in liquid CO2 trucking cost ($/yr, user-specified output dollar year). 
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    input_dollar_year = df_co2['Output Dollar Year (User Input)'].values[0]
    
    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )
    
    # find nearest low and high values to given CO2 flowrate
    x = CO2_flow_kt_per_yr
    xs = df_co2['Size (kt-CO2/y)'].unique()
    if x < xs.min() or x > xs.max():
        raise ValueError(
            'CO2 flowrate needs to be between {} and {} ktonne/year.'.format(
                xs.min(), xs.max())
            )
    x1 = xs[xs <= x].max()
    x2 = xs[xs >= x].min()
    
    # find nearest low and high values to given transport distance
    y = deliv_dist_mi
    ys = df_co2['Distance (mi)'].unique()
    if y < ys.min() or y > ys.max():
        raise ValueError(
            'Transport distance needs to be between {} and {} miles.'.format(
                ys.min(), ys.max())
            )
    y1 = ys[ys <= y].max()
    y2 = ys[ys >= y].min()
    
    # find transport costs corresponding to nearest CO2 flowrates and distances
    z11 = df_co2['Total ($/t-CO2 gross)'].loc[(
        df_co2['Size (kt-CO2/y)'] == x1
        ) & (
        df_co2['Distance (mi)'] == y1
        )].values[0]
    z12 = df_co2['Total ($/t-CO2 gross)'].loc[(
        df_co2['Size (kt-CO2/y)'] == x1
        ) & (
        df_co2['Distance (mi)'] == y2
        )].values[0]
    z21 = df_co2['Total ($/t-CO2 gross)'].loc[(
        df_co2['Size (kt-CO2/y)'] == x2
        ) & (
        df_co2['Distance (mi)'] == y1
        )].values[0]
    z22 = df_co2['Total ($/t-CO2 gross)'].loc[(
        df_co2['Size (kt-CO2/y)'] == x2
        ) & (
        df_co2['Distance (mi)'] == y2
        )].values[0]
    
    # interpolate transport cost ($/tCO2 gross, input dollar year)
    if (x1 == x2) and (y1 == y2):
        z = z11
    elif x1 == x2:
        z = (z11 * (y2 - y) + z22 * (y - y1)) / (y2 - y1) 
    elif y1 == y2:
        z = (z11 * (x2 - x) + z22 * (x - x1)) / (x2 - x1) 
    else:
        z = (
            z11 * (x2 - x) * (y2 - y) + \
            z21 * (x - x1) * (y2 - y) + \
            z12 * (x2 - x) * (y - y1) + \
            z22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1))
    
    # calculate transport cost ($/tCO2 gross, output dollar year)
    liq_CO2_trucking_cost_usd_per_tCO2 = z * dollar_year_multiplier
    
    # calculate annual transport cost ($/yr, output dollar year)
    liq_CO2_trucking_cost_usd_per_yr = \
        liq_CO2_trucking_cost_usd_per_tCO2 * tonne_per_kt * \
        CO2_flow_kt_per_yr
    
    return liq_CO2_trucking_cost_usd_per_tCO2, \
        liq_CO2_trucking_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: STORAGE

# ----------------------------------------------------------------------------
# function: compressed hydrogen terminal storage sizing

# TODO: revisit storage tank capacity (kg) and usable tank capacity fraction
# NOTE: calculated in HDSAM V3.1 from hydrogen compressibility (function of 
# storage temperature and pressure)

def GH2_terminal_storage_size(
        H2_flow_kg_per_day,
        stor_amt_days = 0.25,
        stor_tank_capacity_kg = 20.3,
        stor_tank_usable_capacity_frac = 0.5
        ): 
    """Calculate compressed hydrogen terminal storage total capacity (kg) and number of storage tanks required. Outputs validated using HDSAM V3.1.
    
    Parameters
    ----------
    H2_flow_kg_per_day : float
        Hydrogen mass flowrate through compressed hydrogen terminal (kg H2/day).
    stor_amt_days : float, optional
        Desired amount of hydrogen stored at compressed hydrogen terminal (days). Default: 0.25 days.
    stor_tank_capacity_kg : float, optional
        Storage tank design capacity (kg) at compressed hydrogen terminal. Default: 20.3 kg (hydrogen @ 25 deg.C, 400 atm; cylinder outside diamter 16 inches, length 30 feet, wall thickness 1.429 inches).
    stor_tank_usable_capacity_frac : float, optional
        Storage tank usable capacity (% of design capacity) at compressed hydrogen terminal. Default: 50% (hydrogen @ 25 deg.C, 400 atm).
    
    Returns
    -------
    stor_tot_capacity_kg
        Total storage capacity required (kg) at compressed hydrogen terminal. 
    num_tanks
        Number of storage tanks required at compressed hydrogen terminal.    
    """
    # calculate total hydrogen storage amount required (kg) at compressed 
    # hydrogen terminal
    stor_amt_kg = H2_flow_kg_per_day * stor_amt_days
        
    # calculate number of storage tanks required at compressed 
    # hydrogen terminal
    num_tanks = math.ceil(
        stor_amt_kg / stor_tank_capacity_kg / \
        stor_tank_usable_capacity_frac
        )
        
    # calculate total storage capacity required (kg) at compressed
    # hydrogen terminal
    stor_tot_capacity_kg = stor_tank_capacity_kg * num_tanks
    
    return stor_tot_capacity_kg, \
        num_tanks

# ----------------------------------------------------------------------------
# function: liquid hydrogen terminal storage sizing

# NOTE: for now, ignore: (a) production plant outages and seasonal peaks 
# (storage at terminal is used to handle daily demand fluctuations only and 
# simply scales with terminal hydrogen flowrate and number days of storage
# required); (b) hydrogen boil-off (0.03% of storage capacity in HDSAM V3.1)

# NOTE: HDSAM V3.1 seems to suggest that liquid hydrogen storage tanks 
# (spheres) can be sized continuously to be smaller than their 
# specified design capacity (termed "maximum volume of single sphere").
# This is different from compressed hydrogen storage tanks, which come in 
# at the specified discrete design capacity in HDSAM V3.1 ("storage cylinder 
# capacity"). For now, follow HDSAM's approach for both compressed and liquid
# hydrogen terminal storage, but this may be worth revisitng, as liquid 
# hydrogen storage cost does not seem to decrease quickly with storage tank
# volume. Note that when terminal storage is assumed to meet only
# daily demand, the maximum volume of the liquid hydrogen storage vessel 
# specified in HDSAM V3.1 (11,000 m^3) is way more than what is needed, 
# e.g., almost 10x the required storage capacity in the case of 
# Sacramento @ 1250 kg/day refueling station capacity.

def LH2_terminal_storage_size(
        H2_flow_kg_per_day,
        stor_amt_days = 1.0,
        stor_tank_max_capacity_cu_m = 11000.0,
        stor_tank_usable_capacity_frac = 0.95
        ): 
    """Calculate liquid hydrogen terminal storage design capacity per tank (m^3) and number of storage tanks required. Outputs validated using HDSAM V3.1.
    
    Parameters
    ----------
    H2_flow_kg_per_day : float
        Hydrogen mass flowrate through liquid hydrogen terminal (kg H2/day).
    stor_amt_days : float, optional
        Desired amount of hydrogen stored at liquid hydrogen terminal (days). Default: 1.0 day.
    stor_tank_max_capacity_cu_m : float, optional
        Storage tank maximum design capacity (m^3) at liquid hydrogen terminal. Note that liquid storage tanks can be sized (continuously) under their maximum design capacity per HDSAM V3.1. Default: 11,000 m^3.
    stor_tank_usable_capacity_frac : float, optional
        Storage tank usable capacity (% of design capacity) at liquid hydrogen terminal. Default: 95%.
    
    Returns
    -------
    stor_tank_capacity_cu_m
        Design capacity of each storage tank (m^3) at liquid hydrogen terminal. 
    num_tanks
        Number of storage tanks required at liquid hydrogen terminal.    
    """
    # calculate total hydrogen storage amount required (kg) at liquid 
    # hydrogen terminal
    stor_amt_kg = H2_flow_kg_per_day * stor_amt_days
        
    # calculate total hydrogen storage amount required (m^3), or "storage 
    # vessel water volume" (HDSAM V3.1 term)
    stor_amt_cu_m = \
        stor_amt_kg / dens_liq_H2_kg_per_cu_m
    
    # calculate number of storage tanks (spheres) required at liquid 
    # hydrogen terminal
    num_tanks = math.ceil(
        stor_amt_cu_m / stor_tank_max_capacity_cu_m / \
        stor_tank_usable_capacity_frac
        )
        
    # calculate volume (m^3) of each storage sphere
    # NOTE: liquid hydrogen storage spheres can be sized continuously under
    # the maximum volume of a single sphere specified in HDSAM V3.1
    stor_tank_capacity_cu_m = \
        stor_amt_cu_m / stor_tank_usable_capacity_frac / \
        num_tanks
    
    return stor_tank_capacity_cu_m, \
        num_tanks

# ----------------------------------------------------------------------------
# function: liquid hydrogen station cryogenic storage sizing

# TODO: incorporate hydrogen boil-off (0.3%) in design tank capacity
# TODO: revisit station capacity
# NOTE: For now, the target station capacity (kg/day) parameter represents 
# both average and "design" (peak or maximum) capacity. Once the average 
# and design capacities are differentiated in the model, the station capacity 
# to be used for station storage sizing should be the design capacity.

def LH2_station_cryo_storage_size(
        stn_capacity_kg_per_day,
        truck_load_kg,
        stor_tank_usable_capacity_frac = 0.95
        ): 
    """Calculate liquid hydrogen refueling station cryogenic storage capacity (kg), with capacity subject to available (discrete) tank capacities. Outputs validated using HDSAM V3.1.
    
    Parameters
    ----------
    stn_capacity_kg_per_day : float
        Hydrogen refueling station capacity (kg/day/station). Design capacity to meet peak demand.
    truck_load_kg : float
        Liquid hydrogen truck load (kg). Technically slightly higher than delivered capacity, which incorporates hydrogen boil-off. For simplicity, can use delivered capacity. 
    stor_tank_usable_capacity_frac : float, optional
        Storage tank usable capacity (% of design capacity) at liquid hydrogen refueling station. Default: 95%.
    
    Returns
    -------
    stor_tot_capacity_kg
        Total capacity of cryogenic storage tank (kg) required at liquid hydrogen refueling station, subject to (discrete) available tank capacities.
    """
    # calculate total hydrogen storage amount required (kg) at liquid 
    # hydrogen refueling station 
    # (= "desired liquid cryogenic tank capacity" in HDSAM V3.1)
    if stn_capacity_kg_per_day < 1/16 * truck_load_kg:
        stor_amt_kg = 1/4 * truck_load_kg
    elif stn_capacity_kg_per_day < 1/9 * truck_load_kg:
        stor_amt_kg = 1/3 * truck_load_kg
    elif stn_capacity_kg_per_day < 1/4 * truck_load_kg:
        stor_amt_kg = 1/2 * truck_load_kg
    elif stn_capacity_kg_per_day <= truck_load_kg:
        stor_amt_kg = truck_load_kg
    else:
        stor_amt_kg = 2 * truck_load_kg
                
    # calculate desired design liquid cryogenic tank capacity (gallon)
    # (= "design liquid cryogenic tank capacity" in HDSAM V3.1)
    design_stor_tank_capacity_gal = \
        stor_amt_kg / stor_tank_usable_capacity_frac / \
        dens_liq_H2_kg_per_cu_m * \
        liter_per_cu_m / liter_per_gal
        
    # calculate total liquid cryogenic tank capacity (gallon), subject to
    # available tank capacities
    # (= "available liquid cryogenic tank capacity" in HDSAM V3.1)
    if design_stor_tank_capacity_gal < 1500.0:
        stor_tot_capacity_gal = 1500.0
    elif (design_stor_tank_capacity_gal - 3000.0) < 100.0:
        stor_tot_capacity_gal = 3000.0
    elif (design_stor_tank_capacity_gal - 6000.0) < 200.0:
        stor_tot_capacity_gal = 6000.0
    elif (design_stor_tank_capacity_gal - 9000.0) < 300.0:
        stor_tot_capacity_gal = 9000.0
    elif (design_stor_tank_capacity_gal - 15000.0) < 400.0:
        stor_tot_capacity_gal = 15000.0
    elif (design_stor_tank_capacity_gal - 20000.0) < 500.0:
        stor_tot_capacity_gal = 20000.0
    elif (design_stor_tank_capacity_gal - 30000.0) < 500.0:
        stor_tot_capacity_gal = 30000.0
    else:
        stor_tot_capacity_gal = 40000.0
        
    # calculate total liquid cryogenic tank capacity (kg)
    stor_tot_capacity_kg = \
        stor_tot_capacity_gal * \
        liter_per_gal / liter_per_cu_m * \
        dens_liq_H2_kg_per_cu_m
        
    return stor_tot_capacity_kg

# ----------------------------------------------------------------------------
# function: refueling station cascade storage sizing (applies to all pathways)

# NOTE: For now, the target station capacity (kg/day) parameter represents 
# both average and "design" (peak or maximum) capacity. Once the average 
# and design capacities are differentiated in the model, the station capacity 
# to be used for station storage sizing should be the design capacity.

def station_cascade_storage_size(
        stn_capacity_kg_per_day,
        casc_stor_size_frac
        ): 
    """Calculate refueling station cascade storage capacity (kg).
    
    NOTE: Station capacity in HDSAM V3.1 for cascade sizing = station *maximum* daily capacity (even though the variable is called "Cascade Size as a Percent of Average Daily Demand" in HDSAM V3.1).Cascade size as a fraction of station daily demand is optimized in HDSAM V3.1; number of cascade systems is an integer and has a minimum of one. For now, model cascade size (% of station capacity) as an input parameter with value informed by HDSAM V3.1 results.
    
    Parameters
    ----------
    stn_capacity_kg_per_day : float
        Hydrogen refueling station capacity (kg/day/station). Design capacity to meet peak demand.
    casc_stor_size_frac : float
        Cascade storage size (fraction of station capacity).
    
    Returns
    -------
    stor_tot_capacity_kg
        Total cascade storage capacity (kg) required at refueling station.
    """
    # calculate cascade storage size required (kg) at refueling station
    stor_tot_capacity_kg = stn_capacity_kg_per_day * casc_stor_size_frac
        
    return stor_tot_capacity_kg

# ----------------------------------------------------------------------------
# function: compressed hydrogen terminal storage installed and O&M costs

# TODO: consider using formula on "Compressed Gas H2 Terminal" tab? ($1200/kg 
# in 2016 dollars)

def GH2_terminal_storage_fixed_costs(    
        stor_tot_capacity_kg, 
        output_dollar_year
        ):
    """Calculate compressed hydrogen terminal storage installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    NOTE: Formula for compressed hydrogen storage cost on the "Compressed Gas H2 Terminal" tab in HDSAM V3.1 does not seem to match any of the storage cost equations on the "Cost Data" tab (taking into account dollar year conversion), but is close to formula for "Low Pressure GH2 Storage System Costs". For now, use equation for "Low Pressure GH2 Storage System Costs" (HDSAM V3.1 "Cost Data" tab).
    
    Parameters
    ----------
    stor_tot_capacity_kg : float
        Total storage capacity required (kg) at compressed hydrogen terminal. 
    output_dollar_year : int
        Dollar year of calculated costs.
    
    Returns
    -------
    stor_inst_cost_usd
        Compressed hydrogen terminal storage installed cost ($, user-specified output dollar year). 
    stor_om_cost_usd_per_yr
        Compressed hydrogen terminal storage annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for low-pressure GH2 storage systems)
    input_dollar_year = 2013

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate compressed hydrogen terminal storage uninstalled 
    # cost ($, output dollar year) 
    stor_uninst_cost_usd = \
        1000.0 * stor_tot_capacity_kg * dollar_year_multiplier
    
    # calculate compressed hydrogen terminal storage installed 
    # cost ($, output dollar year)
    stor_inst_cost_usd = stor_uninst_cost_usd * inst_factor
    
    # calculate compressed hydrogen terminal storage annual O&M cost 
    # ($/yr, output dollar year)
    # HDSAM V3.1: 1% of storage installed cost ("remainder of facility")
    stor_om_cost_usd_per_yr = 0.01 * stor_inst_cost_usd
    
    return stor_inst_cost_usd, \
        stor_om_cost_usd_per_yr, \
        output_dollar_year

# ----------------------------------------------------------------------------
# function: liquid hydrogen terminal storage installed and O&M costs

# TODO: revisit cost equation 
# using equation on "Cost Data" tab, calculated liquid hydrogen terminal 
# storage cost is ~1/2 the cost on "Liquid H2 Terminal" tab (different 
# formula)

def LH2_terminal_storage_fixed_costs(
        stor_tank_capacity_cu_m, 
        num_tanks,
        output_dollar_year
        ):
    """Calculate liquid hydrogen terminal storage installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    For now, use equation for "LH2 Storage Costs" (HDSAM V3.1 "Cost Data" tab). 
    
    NOTE: the equation for "LH2 Storage Costs" on the "Cost Data" tab is different from the formula used for liquid storage cost calculation on the "Liquid H2 Terminal" tab, but the former is consistent with the equation in Nexant et al., "Interim Report", Figure 2-44. 
        
    Parameters
    ----------
    stor_tank_capacity_cu_m : float
        Design capacity of each storage tank (m^3) at liquid hydrogen terminal. 
    num_tanks : int
        Number of storage tanks required at liquid hydrogen terminal.    
    output_dollar_year : int
        Dollar year of calculated costs.
    
    Returns
    -------
    stor_inst_cost_usd
        Liquid hydrogen terminal storage installed cost ($, user-specified output dollar year). 
    stor_om_cost_usd_per_yr
        Liquid hydrogen terminal storage annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for "LH2 Storage Costs" on "Cost Data" tab)
    input_dollar_year = 2005

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3
    
    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate liquid hydrogen terminal storage uninstalled 
    # cost ($, output dollar year) 
    stor_uninst_cost_usd = num_tanks * (
        -0.1674 * stor_tank_capacity_cu_m**2 + \
        2064.6 * stor_tank_capacity_cu_m + 977886.0
        ) * dollar_year_multiplier
    
    # calculate liquid hydrogen terminal storage installed 
    # cost ($, output dollar year)
    stor_inst_cost_usd = stor_uninst_cost_usd * inst_factor
    
    # calculate liquid hydrogen terminal storage annual O&M cost 
    # ($/yr, output dollar year)
    # HDSAM V3.1: 1% of storage installed cost
    stor_om_cost_usd_per_yr = 0.01 * stor_inst_cost_usd
    
    return stor_inst_cost_usd, \
        stor_om_cost_usd_per_yr, \
        output_dollar_year
        
# ----------------------------------------------------------------------------
# function: liquid hydrogen refueling station cryogenic storage installed 
# and O&M costs

def LH2_station_cryo_storage_fixed_costs(
        stor_tot_capacity_kg, 
        output_dollar_year
        ):
    """Calculate liquid hydrogen refueling station cryogenic storage installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
            
    Parameters
    ----------
    stor_tot_capacity_kg : float
        Total capacity of cryogenic storage tank (kg) required at liquid hydrogen refueling station.
    output_dollar_year : int
        Dollar year of calculated costs.
    
    Returns
    -------
    stor_inst_cost_usd
        Liquid hydrogen refueling station cryogenic storage installed cost ($, user-specified output dollar year). 
    stor_om_cost_usd_per_yr
        Liquid hydrogen refueling station cryogenic storage annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for "LH2 Storage Costs" on "Cost Data" tab)
    input_dollar_year = 2013

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3
    
    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate liquid hydrogen refueling station cryogenic storage uninstalled 
    # cost ($, output dollar year) 
    stor_uninst_cost_usd = \
        991.89 * stor_tot_capacity_kg**0.6929 * \
        dollar_year_multiplier
        
    # calculate liquid hydrogen refueling station cryogenic storage installed 
    # cost ($, output dollar year)
    stor_inst_cost_usd = stor_uninst_cost_usd * inst_factor
    
    # calculate liquid hydrogen refueling station cryogenic storage annual 
    # O&M cost ($/yr, output dollar year)
    # HDSAM V3.1: 1% of storage installed cost
    stor_om_cost_usd_per_yr = 0.01 * stor_inst_cost_usd
    
    return stor_inst_cost_usd, \
        stor_om_cost_usd_per_yr, \
        output_dollar_year

# ----------------------------------------------------------------------------
# function: refueling station cascade storage installed and O&M costs

def station_cascade_storage_fixed_costs(
        stor_tot_capacity_kg, 
        output_dollar_year
        ):
    """Calculate refueling station cascade storage installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
    
    For now, use equation for "700 Bar Cascade Storage System Costs" (HDSAM V3.1 "Cost Data" tab).
    
    Parameters
    ----------
    stor_tot_capacity_kg : float
        Total cascade storage capacity required (kg) at refueling station.
    output_dollar_year : int
        Dollar year of calculated costs.
    
    Returns
    -------
    stor_inst_cost_usd
        Refueling station cascade storage installed cost ($, user-specified output dollar year). 
    stor_om_cost_usd_per_yr
        Refueling station cascade storage annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify input dollar year 
    # (dollar year in HDSAM V3.1 for cascade storage systems)
    input_dollar_year = 2013

    # specify installation factor 
    # (scales uninstalled cost to installed cost)
    inst_factor = 1.3

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate refueling station cascade storage uninstalled 
    # cost ($, output dollar year) 
    stor_uninst_cost_usd = \
        1800.0 * stor_tot_capacity_kg * dollar_year_multiplier
    
    # calculate refueling station cascade storage installed 
    # cost ($, output dollar year)
    stor_inst_cost_usd = stor_uninst_cost_usd * inst_factor
    
    # calculate refueling station cascade storage annual O&M cost 
    # ($/yr, output dollar year)
    # HDSAM V3.1: 1% of storage installed cost
    stor_om_cost_usd_per_yr = 0.01 * stor_inst_cost_usd
    
    return stor_inst_cost_usd, \
        stor_om_cost_usd_per_yr, \
        output_dollar_year
                
# ----------------------------------------------------------------------------
# function: general-purpose tank storage sizing

def general_tank_storage_size(
        fluid_flow_kg_per_day,
        stor_amt_days,
        fluid_dens_kg_per_cu_m,
        stor_tank_usable_capacity_frac = 0.95
        ):
    """Calculate general-purpose storage tank design capacity per tank (m^3) and number of storage tanks required.
    
    Reference(s): 
    (a) Woods, 2007, "Rules of Thumb Engineering Practice"
    
    Parameters
    ----------
    fluid_flow_kg_per_day : float
        Mass flowrate of fluid to be stored (kg fluid/day).
    stor_amt_days : float
        Desired amount of fluid stored (days). 
    fluid_dens_kg_per_cu_m : float
        Fluid density (kg/m^3).
    stor_tank_usable_capacity_frac : float, optional
        Storage tank usable capacity (% of design capacity). Default: 95% (HDSAM V3.1 for liquid hydrogen).
    
    Returns
    -------
    stor_tank_capacity_cu_m
        Design capacity of each storage tank (m^3). 
    num_tanks
        Number of storage tanks required.    
    """
    # minimum and maximum storage tank capacity (m^3)
    # minimum and maximum = size for which cost scaling
    # is available in Woods, 2007.
    stor_tank_min_capacity_cu_m = 4.0
    stor_tank_max_capacity_cu_m = 70.0

    # calculate total storage amount required (kg) 
    stor_amt_kg = fluid_flow_kg_per_day * stor_amt_days
        
    # calculate total storage amount required (m^3)
    stor_amt_cu_m = stor_amt_kg / fluid_dens_kg_per_cu_m
    
    # calculate number of storage tanks required
    num_tanks = math.ceil(
        stor_amt_cu_m / stor_tank_max_capacity_cu_m / \
        stor_tank_usable_capacity_frac
        )
            
    # calculate volume (m^3) of each storage tank
    # NOTE: assume storage tanks can be sized continuously
    stor_tank_capacity_cu_m = max(
        stor_tank_min_capacity_cu_m, 
        stor_amt_cu_m / stor_tank_usable_capacity_frac / \
        num_tanks
        )
        
    return stor_tank_capacity_cu_m, \
        num_tanks
        
# ----------------------------------------------------------------------------
# function: general-purpose tank storage installed and O&M costs

def general_tank_stor_fixed_costs(
        stor_tank_capacity_cu_m,
        num_tanks,
        output_dollar_year, 
        material = 'carbon steel',
        ):
    """Calculate general-purpose storage tank installed cost ($) and annual O&M cost ($/yr), both in user-specified output dollar year.
            
    Reference(s): 
    (a) Woods, 2007, "Rules of Thumb Engineering Practice"

    Parameters
    ----------
    stor_tank_capacity_cu_m : float
        Design capacity of each storage tank (m^3). 
    num_tanks : int
        Number of storage tanks required.    
    output_dollar_year : int
        Dollar year of calculated costs. 
    material : str
        Tank material. 
        Currently accepted materials: 'carbon steel', 'fiber glass open top', 'rubber-lined', 'lead-lined', 'stainless steel'. 
        Default: 'carbon steel'
                
    Returns
    -------
    stor_tank_inst_cost_usd
        Storage tank installed cost ($, user-specified output dollar year). 
    stor_tank_om_cost_usd_per_yr
        Storage tank annual O&M cost ($/yr, user-specified output dollar year).
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    #------------------------------------------------------------------------#
    # inputs and assumptions
    #------------------------------------------------------------------------#

    # accepted materials
    materials = [
        'carbon steel', 
        'fiber glass open top', 
        'rubber-lined', 
        'lead-lined', 
        'stainless steel', 
        ]
    
    # output error message if user-supplied material is not one of 
    # the accepted materials
    if material not in materials:
        raise ValueError(
            'Invalid material. Accepted materials: {}'.format(materials)
        )
    
    # input CEPCI
    # CEPCI associated with costs in Woods, 2007
    input_cepci = 1000

    # installation factor
    # Woods, 2007: "L+M*" (labor and materials) factor;
    # converts free-on-board (FOB) cost to bare module cost. 
    # "*" denotes the exlusion of instrumentation material and 
    # labor costs.
    # L+M* for small, low pressure tank (vertical cylinder
    # with usual nozzles): 2.3–2.9; 
    # high value for installation of one unit, low value for many units.
    inst_factor = 2.6

    # material cost factor
    mater_factor = 1.0
    if material == 'fiber glass open top':
        mater_factor = 1.6
    if material == 'rubber-lined':
        mater_factor = 1.5
    if material == 'lead-lined':
        mater_factor = 1.6
    if material == 'stainless steel':
        mater_factor = 2.0
    
    # specify reference storage tank capacity (m^3)
    # reference = base size for cost scaling.
    ref_stor_tank_capacity_cu_m = 20.0

    # specify reference FOB cost ($, input CEPCI)
    ref_tank_purc_cost_usd = 14000.0

    #------------------------------------------------------------------------#
    # calculations
    #------------------------------------------------------------------------#

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            output_dollar_year = output_dollar_year, 
            input_cost_index = input_cepci
            )
    
    # calculate storage tank purchase or FOB cost
    # ($, output dollar year) 
    stor_tank_purc_cost_usd = \
        num_tanks * ref_tank_purc_cost_usd * (
        stor_tank_capacity_cu_m / ref_stor_tank_capacity_cu_m
        )**0.71 * mater_factor * dollar_year_multiplier
        
    # calculate storage tank installed cost ($, output dollar year)
    stor_tank_inst_cost_usd = stor_tank_purc_cost_usd * inst_factor

    # calculate storage tank annual O&M cost ($/yr, output dollar year)
    # HDSAM V3.1: 1% of storage installed cost; use for general-purpose tanks
    stor_tank_om_cost_usd_per_yr = 0.01 * stor_tank_inst_cost_usd

    return stor_tank_inst_cost_usd, \
        stor_tank_om_cost_usd_per_yr, \
        output_dollar_year

#%% FUNCTIONS: TERMINAL AND REFUELING STATION COSTS

# function: *non-refueling station* total capital investment
def non_station_total_capital_investment(
        init_cap_inv_usd, 
        input_dollar_year, 
        output_dollar_year = None, 
        indir_cost_perc_override = None
        ):
    """Calculate non-refueling station (e.g., terminal) total capital investment ($) in user-specified output dollar year.

    Total capital investment = installed costs + indirect costs for all station or non-station components. Indirect costs include site preparation, engineering and design, project contingency, one-time licensing fees, upfront permitting costs, and owner's cost.

    Parameters
    ----------
    init_cap_inv_usd : float
        Initial capital investment ($) of non-refueling station (e.g., terminal). Sum of installed costs of components (e.g., compressor, pump).
    input_dollar_year : int
        Input data dollar year.
    output_dollar_year : int, optional
        Dollar year of calculated costs. Default: same as input dollar year.
    indir_cost_perc_override : float, optional
        User-supplied total indirect cost percentage (% initial capital investment). Default: indirect costs from HDSAM V3.1.
        
    Returns
    -------
    tot_cap_inv_usd
        Total capital investment ($, user-specified output dollar year) for non-refueling station.
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify indirect cost percentages (% initial capital investment)
    # indirect costs include site preparation, engineering and design, 
    # project contingency, one-time licensing fees, upfront permitting 
    # costs, owner's cost
    indir_cost_perc_site_prep = 0.05
    indir_cost_perc_eng_design = 0.10
    indir_cost_perc_proj_contingency = 0.10
    indir_cost_perc_licensing = 0.0
    indir_cost_perc_permitting = 0.03
    indir_cost_perc_owner = 0.12 

    # calculate total indirect cost percentage (% initial capital investment)
    tot_indir_cost_perc = \
        indir_cost_perc_site_prep + \
        indir_cost_perc_eng_design + \
        indir_cost_perc_proj_contingency + \
        indir_cost_perc_licensing + \
        indir_cost_perc_permitting + \
        indir_cost_perc_owner
    
    # set indirect cost override to calculated total indirect cost percentage 
    # if user input not provided
    if indir_cost_perc_override == None:
        indir_cost_perc_override = tot_indir_cost_perc
    
    # assume output dollar year is same as input dollar year by default
    if output_dollar_year == None:
        output_dollar_year = input_dollar_year

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate total capital investment ($, output dollar year)
    tot_cap_inv_usd = \
        init_cap_inv_usd * \
        (1 + indir_cost_perc_override) * dollar_year_multiplier

    return tot_cap_inv_usd, \
        output_dollar_year

# ----------------------------------------------------------------------------
# function: *refueling station* total capital investment  
       
def station_total_capital_investment(
        init_cap_inv_usd, 
        input_dollar_year, 
        output_dollar_year = None, 
        indir_cost_perc_override = None
        ):
    """Calculate refueling station total capital investment ($) in user-specified output dollar year.

    Total capital investment = installed costs + indirect costs for all station or non-station components. Indirect costs include site preparation, engineering and design, project contingency, one-time licensing fees, upfront permitting costs. Note: no owner's cost (difference from non-station indirect costs).

    Parameters
    ----------
    init_cap_inv_usd : float
        Initial capital investment ($) of refueling station. Sum of installed costs of components (e.g., compressor, pump).
    input_dollar_year : int
        Input data dollar year.
    output_dollar_year : int, optional
        Dollar year of calculated costs. Default: same as input dollar year.
    indir_cost_perc_override : float, optional
        User-supplied total indirect cost percentage (% initial capital investment). Default: indirect costs from HDSAM V3.1.
        
    Returns
    -------
    tot_cap_inv_usd
        Total capital investment ($, user-specified output dollar year) for refueling station.
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify indirect cost percentages (% initial capital investment)
    # indirect costs include site preparation, engineering and design, 
    # project contingency, one-time licensing fees, upfront permitting costs
    # NOTE: no owner's cost (difference from non-station indirect costs)
    indir_cost_perc_site_prep = 0.05
    indir_cost_perc_eng_design = 0.10
    indir_cost_perc_proj_contingency = 0.05
    indir_cost_perc_licensing = 0.0
    indir_cost_perc_permitting = 0.03
    
    # calculate total indirect cost percentage (% initial capital investment)
    tot_indir_cost_perc = \
        indir_cost_perc_site_prep + \
        indir_cost_perc_eng_design + \
        indir_cost_perc_proj_contingency + \
        indir_cost_perc_licensing + \
        indir_cost_perc_permitting
    
    # set indirect cost override to calculated total indirect cost percentage
    # if user input not provided
    if indir_cost_perc_override == None:
        indir_cost_perc_override = tot_indir_cost_perc
       
    # assume output dollar year is same as input dollar year by default
    if output_dollar_year == None:
        output_dollar_year = input_dollar_year

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate total capital investment ($, output dollar year)
    tot_cap_inv_usd = \
        init_cap_inv_usd * \
        (1 + indir_cost_perc_override) * dollar_year_multiplier

    return tot_cap_inv_usd, \
        output_dollar_year
        
# ----------------------------------------------------------------------------
# function: other operation and maintenance (O&M) costs for refueling station 
# or non-refueling station

def other_om_cost(
        tot_cap_inv_usd,
        input_dollar_year,
        output_dollar_year = None, 
        om_cost_perc_override = None
        ):
    """Calculate other O&M costs ($/yr) for refueling station or non-refueling station (e.g., terminal, liquefier) in user-specified output dollar year.
    
    Other O&M costs include insurance, property taxes, licensing and permit. These costs are functions of total capital investment of refueling station or non-refueling station.

    Other O&M costs do *not* include operation, maintenance, and repair costs for individual components (e.g., compressor, liquefier). These costs are functions of equipment installed cost and are calculated in the cost function for each component.

    Parameters
    ----------
    init_cap_inv_usd : float
        Initial capital investment ($) of refueling station. Sum of installed costs of components (e.g., compressor, pump).
    input_dollar_year : int
        Input data dollar year.
    output_dollar_year : int, optional
        Dollar year of calculated costs. Default: same as input dollar year.
    om_cost_perc_override : float, optional
        User-supplied other O&M cost percentage (% total capital investment). Default: other O&M costs from HDSAM V3.1.

    Returns
    -------
    tot_other_om_cost_usd_per_yr
        Total other O&M costs ($/yr, user-specified output dollar year) for refueling station or non-refueling station.
    output_dollar_year
        User-specified output dollar year, for sanity check.        
    """
    # specifiy "other" O&M cost percentages (% total capital investment)
    # "other" O&M costs include insurance, property taxes, licensing and 
    # permits   
    other_om_cost_perc_insurance = 0.01
    other_om_cost_perc_property = 0.01
    other_om_cost_perc_licensing = 0.001

    # calculate total "other" O&M percentage (% total capital investment)
    tot_other_om_cost_perc = \
        other_om_cost_perc_insurance + \
        other_om_cost_perc_property + \
        other_om_cost_perc_licensing
        
    # set O&M cost percentage override to calculated total O&M cost percentage 
    # if user input not provided
    if om_cost_perc_override == None:
        om_cost_perc_override = tot_other_om_cost_perc
        
    # assume output dollar year is same as input dollar year by default
    if output_dollar_year == None:
        output_dollar_year = input_dollar_year

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate "other" annual O&M costs ($/yr, output dollar year)    
    tot_other_om_cost_usd_per_yr = \
        tot_cap_inv_usd * om_cost_perc_override * \
        dollar_year_multiplier
    
    return tot_other_om_cost_usd_per_yr, \
        output_dollar_year

# ----------------------------------------------------------------------------
# function: *non-refueling station* labor cost

def non_station_labor_cost(
        H2_flow_kg_per_day, 
        output_dollar_year, 
        non_station_labor_rate_usd_per_hr = 27.51, 
        input_dollar_year = 2014
        ):
    """Calculate non-refueling station annual labor cost ($/yr) in user-specified output dollar year. Outputs validated using HDSAM V3.1.
    
    Labor cost includes overhead and general and administrative (G&A). Calculations apply to hydrogen terminal and liquefier.
    
    Parameters
    ----------
    H2_flow_kg_per_day : float
        Hydrogen mass flowrate through non-refueling station (kg H2/day).
    output_dollar_year : int
        Dollar year of calculated costs.    
    non_station_labor_rate_usd_per_hr : float, optional
        Non-refueling station hourly labor rate ($/hr). Default: data from HDSAM V3.1 "Cost Data" tab.
    input_dollar_year: int, optional
        Input data dollar year. Default: data from HDSAM V3.1 "Cost Data" tab.
        
    Returns
    -------
    non_station_labor_cost_usd_per_yr
        Non-refueling station annual labor cost ($/yr, user-specified output dollar year). 
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify overhead and G&A percentage (% unburdened labor cost)
    # HDSAM V3.1: 50% of unburdened labor cost for terminal and liquefier
    non_station_labor_cost_overhead_perc = 0.5

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate annual non-refueling station labor hours (hr/yr)
    non_station_labor_hr_per_yr = \
            17520 * (H2_flow_kg_per_day / 1.0e5)**0.25
    
    # calculate annual unburdened non-refueling station labor cost 
    # ($/yr, output dollar year)
    unburdened_non_station_labor_cost_usd_per_yr = \
        non_station_labor_hr_per_yr * \
        non_station_labor_rate_usd_per_hr * dollar_year_multiplier

    # calculate annual non-refueling station labor cost, including overhead 
    # and G&A ($/yr, output dollar year)
    non_station_labor_cost_usd_per_yr = \
        unburdened_non_station_labor_cost_usd_per_yr * \
        (1 + non_station_labor_cost_overhead_perc)
    
    return non_station_labor_cost_usd_per_yr, \
        output_dollar_year
        
# ----------------------------------------------------------------------------
# function: *refueling station* labor cost

def station_labor_cost(
        H2_flow_kg_per_day, 
        output_dollar_year, 
        station_labor_rate_usd_per_hr = 10.44, 
        input_dollar_year = 2014
        ):
    """Calculate refueling station annual labor cost ($/yr) in user-specified output dollar year. Outputs validated using HDSAM V3.1.
    
    Labor cost includes overhead and general and administrative (G&A).
    
    Parameters
    ----------
    H2_flow_kg_per_day : float
        Hydrogen mass flowrate through refueling station (kg H2/day).
    output_dollar_year : int
        Dollar year of calculated costs.    
    station_labor_rate_usd_per_hr : float, optional
        Refueling station hourly labor rate ($/hr). Default: data from HDSAM V3.1 "Cost Data" tab.
    input_dollar_year: int, optional
        Input data dollar year. Default: data from HDSAM V3.1 "Cost Data" tab.
        
    Returns
    -------
    station_labor_cost_usd_per_yr
        Refueling station annual labor cost ($/yr, user-specified output dollar year). 
    output_dollar_year
        User-specified output dollar year, for sanity check.
    """
    # specify overhead and G&A percentage (% unburdened labor cost)
    # HDSAM V3.1: 20% of unburdened labor cost for refueling station
    station_labor_cost_overhead_perc = 0.2

    # calculate conversion (multiplier) from input dollar year to 
    # output dollar year
    dollar_year_multiplier = \
        dollar_year_conversion(
            input_dollar_year = input_dollar_year, 
            output_dollar_year = output_dollar_year
            )    

    # calculate annual refueling station labor hours (hr/yr)
    # Use equation on "Refueling Station - Gaseous H2" tab (same as 
    # "Refueling Station - Liquid H2" tab) for "labor required (hrs/year)" 
    # in HDSAM V3.1.
    # NOTE: Base annual labor (3252.15 hr/yr, calculated, see next note) on 
    # "Refueling Station - Gaseous H2" tab differs from "Cost Data" tab 
    # (3285 hr/yr) in HDSAM V3.1.
    # Nexant et al., "Interim Report", p. 2-20: 18 hours of operation per 
    # day, 365 days of operation per year, 1.5 average persons in snack 
    # store, 33% of snack store labor associated with fuel dispensing.  
    # --> annual labor hours allocated to fuel dispensing = 3252 hr/yr.
    station_labor_hr_per_yr = \
        18 * 365 * 1.5 * 0.33 * (H2_flow_kg_per_day / 1050)**0.25
    
    # calculate annual unburdened refueling station labor cost 
    # ($/yr, output dollar year)
    unburdened_station_labor_cost_usd_per_yr = \
        station_labor_hr_per_yr * station_labor_rate_usd_per_hr * \
        dollar_year_multiplier

    # calculate annual refueling station labor cost, including overhead 
    # and G&A ($/yr, output dollar year)
    station_labor_cost_usd_per_yr = \
        unburdened_station_labor_cost_usd_per_yr * \
        (1 + station_labor_cost_overhead_perc)
    
    return station_labor_cost_usd_per_yr, \
        output_dollar_year

#%% TECHNOECONOMIC ANALYSIS FOR VARIOUS HYDROGEN DELIVERY PATHWAYS

def calcs(
        dict_input_params = {
            'run #' : 0,
            'output dollar year' : 2022,
            'target station capacity (kg/day)' : 1000.0,
            'target number of stations' : 10,
            'one-way delivery distance (mile)' : 100.0,
            'electricity cost ($/kWh)' : 0.1709,
            'diesel cost ($/gallon)' : 6.028,
            'electricity emission factor (kg CO2/kWh)' : 0.228,
            'diesel emission factor (kg CO2/gallon)' : 10.180,
            'hydrogen prod. emission factor (kg CO2-eq/kg)' : 0.0,
            'formic acid prod. emission factor (kg CO2-eq/kg)' : 0.0,            
            'hydrogen purchase cost ($/kg)' : 0.31,
            'formic acid purchase cost ($/kg)' : 1.0,
            'formic acid production pathway' : 'electro', 
            'hydr. reaction temperature (K)' : 366.15,
            'hydr. reaction pressure (bar)' : 105.0,
            'hydr. reaction yield' : 1.0,
            'hydr. reactor volume (m^3)' : 1.0,
            'number of hydr. reactors' : 1,
            'hydr. catalyst amount (kg)' : 53.0,
            'hydr. catalyst cost ($/kg)' : 5450.0,
            'hydr. catalyst lifetime (yr)' : 1.0,
            'hydr. reactor energy (unit TBD)' : 0.0,
            'hydr. separator energy (unit TBD)' : 0.0,
            'CO2 electrolyzer purchase cost ($/m^2)' : 5250.0,
            'terminal formic acid storage amount (days)' : 0.25,
            'terminal compressed hydrogen storage amount (days)': 0.25,
            'terminal liquid hydrogen storage amount (days)': 1.0,
            'dehydr. reaction temperature (K)' : 300.0,
            'dehydr. reaction pressure (bar)' : 1.013,
            'dehydr. reaction yield' : 0.9999,
            'dehydr. reactor volume (m^3)' : 0.073,
            'number of dehydr. reactors' : 1,
            'dehydr. catalyst amount (kg)' : 9.65,
            'dehydr. catalyst cost ($/kg)' : 3500,
            'dehydr. catalyst lifetime (yr)' : 1.0,
            'dehydr. reactor energy (unit TBD)' : 0.0,
            'dehydr. gas/liquid separator energy (unit TBD)' : 0.0,
            'station formic acid storage amount (days)' : 1.0,
            }, 
        save_csv = False,
        output_folder = 'outputs'
        ):
    """Run S2A systems technoeconomic analysis for various delivery pathways.

    Parameters
    ----------
    dict_input_params : dict
        Dictionary containing one set of input parameters for one run.
    save_csv : bool, default False
        If True, save output csv file. Otherwise return total hydrogen cost ($/kg) by pathway and output dataframe.
    output_folder : str, default 'outputs'
        Path for saving output .csv files.
        
    Returns
    -------
    df_output
        Return dataframe containing input parameters and results (energy consumption, costs, etc.). 
    FAH2_tot_H2_cost_usd_per_kg
        Return total $/kg H2 costs for formic acid delivery pathway ("FAH2"). 
    GH2_tot_H2_cost_usd_per_kg
        Return total $/kg H2 costs for compressed gaseous hydrogen delivery pathway ("GH2").
    LH2_tot_H2_cost_usd_per_kg
        Return total $/kg H2 costs for liquid hydrogen delivery pathway ("LH2").
    """
    #%% ASSIGN INPUT PARAMETERS
    
    # run number
    run_num = dict_input_params[
        'run #'
        ]
    
    # dollar year of calculated costs
    output_dollar_year = dict_input_params[
        'output dollar year'
        ]

    # target hydrogen refueling station capacity (kg/day/station)
    target_stn_capacity_kg_per_day = dict_input_params[
        'target station capacity (kg/day)'
        ]
    
    # target number of hydrogen refueling stations
    target_num_stns = dict_input_params[
        'target number of stations'
        ]
    
    # delivery distance (mile), one-way
    deliv_dist_mi_ow = dict_input_params[
        'one-way delivery distance (mile)'
        ]
    
    # electricity cost ($/kWh)
    elec_cost_usd_per_kWh = dict_input_params[
        'electricity cost ($/kWh)'
        ]

    # diesel cost ($/gallon)
    diesel_cost_usd_per_gal = dict_input_params[
        'diesel cost ($/gallon)'
        ]

    # electricity emission factor (kg CO2/kWh)
    elec_ghg_kg_CO2_per_kWh = dict_input_params[
        'electricity emission factor (kg CO2/kWh)'
        ]
    
    # diesel emission factor (kg CO2/gallon)
    diesel_ghg_kg_CO2_per_gal = dict_input_params[
        'diesel emission factor (kg CO2/gallon)'
        ]
    
    # hydrogen production emission factor (kg CO2-eq/kg H2)
    H2_prod_ghg_kg_CO2_per_kg = dict_input_params[
        'hydrogen prod. emission factor (kg CO2-eq/kg)'
        ]
    
    # formic acid production emission factor (kg CO2-eq/kg formic acid)
    FA_prod_ghg_kg_CO2_per_kg = dict_input_params[
        'formic acid prod. emission factor (kg CO2-eq/kg)'
        ]
    
    # hydrogen purchase costs ($/kg H2)
    purc_cost_H2_usd_per_kg = dict_input_params[
        'hydrogen purchase cost ($/kg)'
        ]

    # formic acid purchase cost ($/kg formic acid)
    purc_cost_FA_usd_per_kg = dict_input_params[
        'formic acid purchase cost ($/kg)'
        ]
    
    # formic acid production pathway
    # thermocatalytic, electrolytic, or purchase
    # 'thermo', 'electro', or 'purchase'
    # purchase = no CO2 recycling or hydrogenation costs
    FA_prod_pathway = dict_input_params[
        'formic acid production pathway'
        ]
    
    # hydrogenation reaction temperature (K)
    FAH2_hydr_temp_K = dict_input_params[
        'hydr. reaction temperature (K)'
        ]
    
    # hydrogenation reaction pressure (bar)
    FAH2_hydr_pres_bar = dict_input_params[
        'hydr. reaction pressure (bar)'
        ]
    
    # hydrogenation reaction yield
    FAH2_hydr_yield = dict_input_params[
        'hydr. reaction yield'
        ]
    
    # hydrogenation reactor volume (m^3)
    FAH2_hydr_react_vol_cu_m = dict_input_params[
        'hydr. reactor volume (m^3)'
        ]
    
    # number of hydrogenation reactors
    FAH2_num_hydr_reacts = dict_input_params[
        'number of hydr. reactors'
        ]
    
    # hydrogenation calalyst amount required (kg)
    FAH2_hydr_catal_amt_kg = dict_input_params[
        'hydr. catalyst amount (kg)'
        ]
    
    # hydrogenation calalyst cost ($/kg)
    FAH2_hydr_catal_cost_usd_per_kg = dict_input_params[
        'hydr. catalyst cost ($/kg)'
        ]

    # hydrogenation catalyst lifetime (yr)
    FAH2_hydr_catal_life_yr = dict_input_params[
        'hydr. catalyst lifetime (yr)'
        ]
    
    # hydrogenation reactor energy requirement (unit TBD)
    FAH2_hydr_react_energy = dict_input_params[
        'hydr. reactor energy (unit TBD)'
        ]

    # hydrogenation separator energy requirement (unit TBD)
    FAH2_hydr_sep_energy = dict_input_params[
        'hydr. separator energy (unit TBD)'
        ]
    
    # CO2 electrolyer purchase cost ($/m^2)
    electr_purc_cost_usd_per_sq_m = dict_input_params[
        'CO2 electrolyzer purchase cost ($/m^2)'
        ]
    
    # formic acid storage amount at terminal (days)
    FAH2_TML_stor_amt_days = dict_input_params[
        'terminal formic acid storage amount (days)'
        ]    

    # compressed hydrogen storage amount at terminal (days)
    GH2_TML_stor_amt_days = dict_input_params[
        'terminal compressed hydrogen storage amount (days)'
        ]

    # liquid hydrogen storage amount at terminal (days)
    LH2_TML_stor_amt_days = dict_input_params[
        'terminal liquid hydrogen storage amount (days)'
        ]

    # dehydrogenation reaction temperature (K)
    # = inlet temperature to precooling for separator 
    # (pressure swing adsorption, PSA) at LOHC (formic acid) 
    # hydrogen refueling station
    FAH2_dehydr_temp_K = dict_input_params[
        'dehydr. reaction temperature (K)'
        ]
    
    # dehydrogenation reaction pressure (bar)
    # = hydrogen outlet pressure (bar) exiting separator (pressure swing
    # adsorption, PSA) at LOHC (formic acid) hydrogen refueling station
    FAH2_dehydr_pres_bar = dict_input_params[
        'dehydr. reaction pressure (bar)'
        ]
    
    # dehydrogenation reaction yield
    FAH2_dehydr_yield = dict_input_params[
        'dehydr. reaction yield'
        ]
    
    # dehydrogenation reactor volume (m^3)
    FAH2_dehydr_react_vol_cu_m = dict_input_params[
        'dehydr. reactor volume (m^3)'
        ]
    
    # number of dehydrogenation reactors
    FAH2_num_dehydr_reacts = dict_input_params[
        'number of dehydr. reactors'
        ]
    
    # dehydrogenation calalyst amount required (kg)
    FAH2_dehydr_catal_amt_kg = dict_input_params[
        'dehydr. catalyst amount (kg)'
        ]
    
    # dehydrogenation calalyst cost ($/kg)
    FAH2_dehydr_catal_cost_usd_per_kg = dict_input_params[
        'dehydr. catalyst cost ($/kg)'
        ]

    # dehydrogenation catalyst lifetime (yr)
    FAH2_dehydr_catal_life_yr = dict_input_params[
        'dehydr. catalyst lifetime (yr)'
        ]
    
    # dehydrogenation reactor energy requirement (unit TBD)
    FAH2_dehydr_react_energy = dict_input_params[
        'dehydr. reactor energy (unit TBD)'
        ]

    # dehydrogenation gas/liquid separator energy requirement (unit TBD)
    FAH2_dehydr_gas_liq_sep_energy = dict_input_params[
        'dehydr. gas/liquid separator energy (unit TBD)'
        ]
    
    # formic acid storage amount at refueling station (days)
    FAH2_STN_stor_amt_days = dict_input_params[
        'station formic acid storage amount (days)'
        ]   
    
    #%% CHECK INPUTS
    
    # accepted formic acid production pathways
    FA_prod_pathways = [
        'thermo', 
        'electro',
        'purchase'
        ]
    
    # raise error if user-specified pathway is not predefined
    if FA_prod_pathway not in FA_prod_pathways:
        raise ValueError(
            'Check formic acid production pathway. '
            'Currently accepted pathways: "thermo", "electro", "purchase".'
            )    

    #%% CALCULATIONS: CREATE LIST FOR WRITING INPUTS AND RESULTS
    
    # initialize list for writing out key input parameters and results
    list_output = []
    
    # assign variable names for list (= column names for dataframe)
    # items in list:
    # - pathway, e.g., compressed hydrogen
    # - process, e.g., preconditioning
    # - location, e.g., terminal
    # - function, e.g., compression
    # - equipment, e.g., compressor
    # - variable group, e.g., O&M cost
    # - variable name, e.g., labor cost
    # - unit, e.g., $/yr
    # - value
    output_columns = [
        'pathway', 
        'process', 
        'location',
        'function',
        'equipment', 
        'variable group', 
        'variable name', 
        'unit', 
        'value'
    ]

    # ------------------------------------------------------------------------
    # append input parameters to list
    
    for key, value in dict_input_params.items():
        
        # extract unit from dictionary of input parameters
        if key.find('(') == -1:
            unit = '-'
        else:
            unit = key[key.find('(') + 1 : key.find(')')]
        
        # extra variable name from dictionary of input parameters
        var_name = key.split(' (')[0]

        # append input parameter to list        
        list_output.append([
            'all', 
            'all', 
            'all', 
            'all', 
            'all', 
            'input parameter', 
            var_name, 
            unit, 
            value
            ])

    #%% CALCULATIONS: GENERAL (UNIT CONVERSIONS, MASS BALANCE)

    # ------------------------------------------------------------------------
    # unit conversions

    # calculate refueling station hydrogen temperature (K)
    # = inlet temperature to compressor and refrigerator
    # = outlet temperature from PSA refrigerator (precooling)
    STN_H2_temp_K = STN_H2_temp_C + C_to_K

    # calculate refueling station hydrogen dispensing temperature (K)
    # = outlet temperature from compressor refrigerator
    # applies to compressed hydrogen and formic acid pathways
    STN_dispens_temp_K = STN_dispens_temp_C + C_to_K
        
    # calculate compressed hydrogen terminal inlet temperature (K)
    # HDSAM V3.1: hydrogen temperature at terminal (low and ambient
    # temperature)
    GH2_TML_in_temp_K = GH2_TML_in_temp_C + C_to_K

    # calculate LOHC / formic acid terminal inlet temperature (K)
    FAH2_TML_in_temp_K = FAH2_TML_in_temp_C + C_to_K

    # calculate LOHC / formic acid refueling station PSA 
    # operating temperature (K)
    FAH2_STN_psa_temp_K = FAH2_STN_psa_temp_C + C_to_K

    # calculate inlet pressure (bar) to storage and truck loading compressors 
    # at compressed hydrogen terminal
    # (= pressure of hydrogen delivered to terminal in HDSAM V3.1)
    # NOTE: HDSAM V3.1 seems a bit inconsistent with the inlet pressure to 
    # truck loading compressor vs. storage compressor. For example, inlet 
    # pressure to truck loading compressor for *power* calculations = 
    # 200 atm = minimum terminal storage pressure; inlet pressure to truck
    # loading compressor for *energy* calculations = 20 atm = pressure of 
    # hydrogen delivered to (compressed hydrogen) terminal. The flowrates 
    # through storage compressor vs. truck loading compressor also seem to 
    # suggest that all hydrogen loaded to trucks is essentially drawn from 
    # terminal storage. For now, assume "steady state": terminal storage holds
    # a fixed amount of hydrogen (specified as # days of storage), and 
    # hydrogen delivered to the terminal is directly sent to both terminal
    # storage and trucks (i.e., same inlet pressure). 
    GH2_TML_in_pres_bar = \
        GH2_TML_in_pres_atm * Pa_per_atm / Pa_per_bar
        
    # calculate inlet pressure (bar) to hydrogenation reactor at 
    # LOHC / formic acid terminal
    FAH2_TML_in_pres_bar = \
        FAH2_TML_in_pres_atm * Pa_per_atm / Pa_per_bar
        
    # ------------------------------------------------------------------------
    # time needed to fill compressed hydrogen terminal storage (days)
    # in HDSAM V3.1, equal to number of days of hydrogen storage at terminal
    GH2_TML_stor_fill_time_days = GH2_TML_stor_amt_days
        
    # ------------------------------------------------------------------------
    # calculate roundtrip delivery distance (mile)
    deliv_dist_mi_rt = deliv_dist_mi_ow * 2
    
    # ------------------------------------------------------------------------
    # general refueling station mass balance

    # calculate total amount of hydrogen delivered per day (kg H2/day) 
    # to all stations
    tot_H2_deliv_kg_per_day = \
        target_stn_capacity_kg_per_day * target_num_stns
    
    # calculate total amount of hydrogen delivered per hour (kg H2/hr) 
    # to all stations
    tot_H2_deliv_kg_per_hr = \
        tot_H2_deliv_kg_per_day / hr_per_day
    
    # calculate total amount of hydrogen delivered per year (kg H2/yr) 
    # to all stations
    tot_H2_deliv_kg_per_yr = tot_H2_deliv_kg_per_day * day_per_yr
        
    # calculate hydrogen mass flowrate (kg/s) at each refueling station
    STN_H2_flow_kg_per_sec = \
        target_stn_capacity_kg_per_day / hr_per_day / sec_per_hr
        
    # calculate hydrogen molar flowrate (mol/s) at each refueling station
    STN_H2_flow_mol_per_sec = \
        STN_H2_flow_kg_per_sec / molar_mass_H2_kg_per_kmol * mol_per_kmol
        
    # ------------------------------------------------------------------------
    # LOHC / formic acid refueling station mass balance
    
    # calculate amount of formic acid delivered per day (kg formic acid/day)
    # to each station for given refueling station capacity 
    FAH2_STN_FA_flow_kg_per_day = \
        target_stn_capacity_kg_per_day / molar_mass_H2_kg_per_kmol / \
        stoic_mol_H2_per_mol_FA * molar_mass_FA_kg_per_kmol / \
        FAH2_dehydr_yield

    # calculate formic acid mass flowrate (kg/s) at each refueling station
    FAH2_STN_FA_flow_kg_per_sec = \
        FAH2_STN_FA_flow_kg_per_day / hr_per_day / sec_per_hr

    # calculate formic acid molar flowrate (mol/hr) delivered to each station
    FAH2_STN_FA_flow_mol_per_hr = \
        FAH2_STN_FA_flow_kg_per_day / \
        molar_mass_FA_kg_per_kmol * mol_per_kmol / hr_per_day
    
    # calculate inlet molar flowrate (mol/hr) to refueling station separator
    FAH2_STN_psa_in_flow_mol_per_hr = \
        FAH2_STN_FA_flow_mol_per_hr * \
        FAH2_dehydr_yield * (stoic_mol_H2_per_mol_FA + stoic_mol_CO2_per_mol_FA)
    
    # calculate inlet volumetric flowrate (Nm^3/hr) to refueling station 
    # separator
    FAH2_STN_psa_in_flow_norm_cu_m_per_hr = \
        mol_to_norm_cu_m(FAH2_STN_psa_in_flow_mol_per_hr)
        
    # calculate total amount of formic acid delivered per day 
    # (kg/day) to all stations
    tot_FA_deliv_kg_per_day = \
        FAH2_STN_FA_flow_kg_per_day * target_num_stns
    
    # calculate amount of CO2 produced per day (kg/day) at each station
    FAH2_STN_CO2_flow_kg_per_day = \
        FAH2_STN_FA_flow_mol_per_hr / mol_per_kmol * hr_per_day * \
        stoic_mol_CO2_per_mol_FA * molar_mass_CO2_kg_per_kmol

    # calculate amount of CO2 produced per year (ktonne/yr) at each station
    FAH2_STN_CO2_flow_kt_per_yr = \
        FAH2_STN_CO2_flow_kg_per_day / \
        kg_per_tonne / tonne_per_kt * day_per_yr
        
    # ------------------------------------------------------------------------
    # general terminal mass balance

    # calculate hydrogen mass flowrate (kg H2/day) at terminal  
    # (applies to compressed or liquid hydrogen terminals)
    # TODO: incorporate hydrogen losses
    TML_H2_flow_kg_per_day = tot_H2_deliv_kg_per_day
    
    # calculate hydrogen mass flowrate (kg/s) at terminal
    # (applies to compressed or liquid hydrogen terminals)
    TML_H2_flow_kg_per_sec = \
        TML_H2_flow_kg_per_day / hr_per_day / sec_per_hr
    
    # calculate hydrogen molar flowrate (mol/s) at terminal
    # (applies to compressed or liquid hydrogen terminals)
    TML_H2_flow_mol_per_sec = \
        TML_H2_flow_kg_per_sec / molar_mass_H2_kg_per_kmol * mol_per_kmol

    # ------------------------------------------------------------------------
    # liquid hydrogen terminal mass balance liquefier size

    # calculate hydrogen mass flowrate (kg H2/day) into liquid terminal
    # "gross" flowrate before boil-off losses
    LH2_TML_H2_in_flow_kg_per_day = TML_H2_flow_kg_per_day / (
        1 - LH2_TML_stor_boil_off_frac_per_day
        )**max(1.0, LH2_TML_stor_amt_days)
    
    # calculate liquefier size (tonne H2/day)
    liquef_size_tonne_per_day = LH2_TML_H2_in_flow_kg_per_day / kg_per_tonne

    # ------------------------------------------------------------------------
    # LOHC / formic acid terminal mass balance

    # calculate formic acid mass flowrate (kg formic acid/day) at 
    # LOHC / formic acid terminal  
    # TODO: incorporate losses
    FAH2_TML_FA_flow_kg_per_day = tot_FA_deliv_kg_per_day
    
    # calculate formic acid mass flowrate (kg/s) at 
    # LOHC / formic acid terminal
    FAH2_TML_FA_flow_kg_per_sec = \
        FAH2_TML_FA_flow_kg_per_day / hr_per_day / sec_per_hr
    
    # calculate formic acid molar flowrate (mol/s) at 
    # LOHC / formic acid terminal
    FAH2_TML_FA_flow_mol_per_sec = \
        FAH2_TML_FA_flow_kg_per_sec / \
        molar_mass_FA_kg_per_kmol * mol_per_kmol
        
    # calculate hydrogen molar flowrate (mol/s) at 
    # LOHC / formic acid terminal
    FAH2_TML_H2_flow_mol_per_sec = \
        FAH2_TML_FA_flow_mol_per_sec / FAH2_hydr_yield * \
        stoic_mol_H2_per_mol_FA

    # calculate hydrogen mass flowrate (kg/day) at LOHC / formic acid terminal  
    FAH2_TML_H2_flow_kg_per_day = \
        FAH2_TML_H2_flow_mol_per_sec / mol_per_kmol * \
        sec_per_hr * hr_per_day * molar_mass_H2_kg_per_kmol
    
    # ------------------------------------------------------------------------
    # transport mass balance

    # calculate formic acid truck delivered capacity 
    # (kg formic acid/truck-trip)
    # NOTE: delivered capacity is limited by weight if cargo has higher
    # density than "rated" density 
    # ("rated" cargo density calculated from maximum weight and volume) 
    liq_truck_deliv_capacity_kgFA = \
        liq_truck_water_vol_cu_m * \
            liq_truck_usable_capacity_frac * min(
                liq_truck_cargo_dens_kg_per_cu_m,
                dens_FA_kg_per_cu_m
                )
    
    #%% CALCULATIONS: COMPRESSED GASEOUS HYDROGEN DELIVERY ("GH2")
    # PRECONDITIONING @ TERMINAL

    # ------------------------------------------------------------------------
    # production - compressed hydrogen: hydrogen purchase cost
    
    # calculate hydrogen purchase cost ($/yr)   
    GH2_TML_H2_purc_cost_usd_per_yr = \
        purc_cost_H2_usd_per_kg * TML_H2_flow_kg_per_day * \
        day_per_yr

    # calculate hydrogen purchase cost ($/kg H2)   
    GH2_TML_H2_purc_cost_usd_per_kg = \
        GH2_TML_H2_purc_cost_usd_per_yr / tot_H2_deliv_kg_per_yr

    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'O&M cost', 
        'purchase cost', 
        '$/yr', 
        GH2_TML_H2_purc_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'O&M cost', 
        'purchase cost', 
        '$/kg H2', 
        GH2_TML_H2_purc_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # production - compressed hydrogen: emissions of purchased hydrogen

    # calculate emissions of purchased hydrogen (kg CO2-eq/kg H2)
    GH2_TML_H2_purc_ghg_kg_CO2_per_kg = \
        H2_prod_ghg_kg_CO2_per_kg * TML_H2_flow_kg_per_day * \
        day_per_yr / tot_H2_deliv_kg_per_yr
    
    # convert emissions of purchased hydrogen to g CO2-eq/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_TML_H2_purc_ghg_g_CO2_per_MJ = \
        GH2_TML_H2_purc_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'emissions', 
        'upstream emissions', 
        'kg CO2/kg H2', 
        GH2_TML_H2_purc_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'emissions', 
        'upstream emissions', 
        'g CO2/MJ H2 (LHV)', 
        GH2_TML_H2_purc_ghg_g_CO2_per_MJ
        ])

    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # truck loading compressor energy consumption and size
    
    # calculate maximum loading pressure for gas truck (tube trailer) (bar)
    # = loading compressor outlet pressure for *power* calculations
    # HDSAM V3.1: about 100 psi above tube trailer maximum pressure
    gas_truck_max_load_pres_bar = \
        (gas_truck_max_tube_pres_atm * Pa_per_atm + \
         100 * Pa_per_psi) / Pa_per_bar
            
    # calculate loading compressor average outlet pressure (bar)
    # for *energy* calculations
    # log-mean of inlet pressure and maximum truck loading pressure
    GH2_TML_load_compr_avg_out_pres_bar = \
        (gas_truck_max_load_pres_bar - GH2_TML_in_pres_bar) / \
        math.log(gas_truck_max_load_pres_bar / GH2_TML_in_pres_bar)
    
    # calculate loading compressor power (kW) and size (kW/stage)
    GH2_TML_load_compr_tot_power_kW, \
    GH2_TML_load_compr_power_kW_per_stg, \
    GH2_TML_load_compr_num_stgs = \
        compressor_power_and_size(
            out_pres_bar = gas_truck_max_load_pres_bar, 
            in_pres_bar = GH2_TML_in_pres_bar, 
            in_temp_K = GH2_TML_in_temp_K, 
            gas_flow_mol_per_sec = TML_H2_flow_mol_per_sec, 
            compressibility = 1.24
            )
    
    # calculate loading compressor average power (kW)
    # for *energy* calculations
    GH2_TML_load_compr_avg_power_kW, _, _ = \
        compressor_power_and_size(
            out_pres_bar = \
                GH2_TML_load_compr_avg_out_pres_bar, 
            in_pres_bar = GH2_TML_in_pres_bar, 
            in_temp_K = GH2_TML_in_temp_K, 
            gas_flow_mol_per_sec = TML_H2_flow_mol_per_sec, 
            compressibility = 1.24
            )
    
    # calculate loading compressor energy (kWh/kg H2)
    GH2_TML_load_compr_elec_kWh_per_kg = \
        GH2_TML_load_compr_avg_power_kW / tot_H2_deliv_kg_per_hr
    
    # convert compressor energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_TML_load_compr_elec_MJ_per_MJ = \
        GH2_TML_load_compr_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        GH2_TML_load_compr_elec_kWh_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        GH2_TML_load_compr_elec_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # truck loading compressor energy emissions

    # calculate loading compressor energy emissions (kg CO2/kg H2)
    GH2_TML_load_compr_ghg_kg_CO2_per_kg = \
        GH2_TML_load_compr_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert loading compressor energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_TML_load_compr_ghg_g_CO2_per_MJ = \
        GH2_TML_load_compr_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        GH2_TML_load_compr_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        GH2_TML_load_compr_ghg_g_CO2_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # truck loading compressor energy cost
    
    # calculate loading compressor energy cost ($/kg H2)
    GH2_TML_load_compr_elec_cost_usd_per_kg = \
        GH2_TML_load_compr_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate loading compressor energy cost ($/yr)
    GH2_TML_load_compr_elec_cost_usd_per_yr = \
        GH2_TML_load_compr_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        GH2_TML_load_compr_elec_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        GH2_TML_load_compr_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # truck loading compressor installed cost and annual O&M cost

    # calculate loading compressor installed cost ($) and annual O&M 
    # cost ($/yr), both in output dollar year 
    GH2_TML_load_compr_inst_cost_usd, \
    GH2_TML_load_compr_om_cost_usd_per_yr, \
    GH2_TML_load_compr_dollar_year = \
        compressor_fixed_costs(
            compr_power_kW_per_stg = GH2_TML_load_compr_power_kW_per_stg, 
            num_stgs = GH2_TML_load_compr_num_stgs, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate loading compressor O&M cost ($/kg H2)
    GH2_TML_load_compr_om_cost_usd_per_kg = \
        GH2_TML_load_compr_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        GH2_TML_load_compr_om_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        GH2_TML_load_compr_om_cost_usd_per_kg
        ])
            
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # storage *compressor* energy consumption and size
    
    # calculate hydrogen molar flowrate (mol/s) to compressed hydrogen 
    # terminal storage    
    GH2_TML_stor_H2_flow_mol_per_sec = \
        TML_H2_flow_mol_per_sec * \
        GH2_TML_stor_amt_days / \
        GH2_TML_stor_fill_time_days

    # calculate terminal storage maximum pressure (bar)
    # = storage compressor outlet pressure for *power* calculations
    GH2_TML_max_stor_pres_bar = \
        GH2_TML_max_stor_pres_atm * Pa_per_atm / Pa_per_bar

    # calculate storage compressor average outlet pressure (bar) 
    # for *energy* calculations
    # log-mean of inlet pressure and terminal storage maximum pressure
    GH2_TML_stor_compr_avg_out_pres_bar = \
        (GH2_TML_max_stor_pres_bar - GH2_TML_in_pres_bar) / \
        math.log(GH2_TML_max_stor_pres_bar / GH2_TML_in_pres_bar)
    
    # calculate storage compressor power (kW) and size (kW/stage)
    GH2_TML_stor_compr_tot_power_kW, \
    GH2_TML_stor_compr_power_kW_per_stg, \
    GH2_TML_stor_compr_num_stgs = \
        compressor_power_and_size(
            out_pres_bar = GH2_TML_max_stor_pres_bar, 
            in_pres_bar = GH2_TML_in_pres_bar, 
            in_temp_K = GH2_TML_in_temp_K, 
            gas_flow_mol_per_sec = GH2_TML_stor_H2_flow_mol_per_sec, 
            compressibility = 1.13
            )
        
    # calculate compressor average power (kW) 
    # for *energy* calculations
    GH2_TML_stor_compr_avg_power_kW, _, _ = \
        compressor_power_and_size(
            out_pres_bar = \
                GH2_TML_stor_compr_avg_out_pres_bar, 
            in_pres_bar = GH2_TML_in_pres_bar, 
            in_temp_K = GH2_TML_in_temp_K, 
            gas_flow_mol_per_sec = GH2_TML_stor_H2_flow_mol_per_sec, 
            compressibility = 1.13
            )
    
    # calculate storage compressor energy (kWh/kg H2)
    GH2_TML_stor_compr_elec_kWh_per_kg = \
        GH2_TML_stor_compr_avg_power_kW / tot_H2_deliv_kg_per_hr
        
    # convert compressor energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_TML_stor_compr_elec_MJ_per_MJ = \
        GH2_TML_stor_compr_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        GH2_TML_stor_compr_elec_kWh_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        GH2_TML_stor_compr_elec_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # storage *compressor* emissions

    # calculate storage compressor energy emissions (kg CO2/kg H2)
    GH2_TML_stor_compr_ghg_kg_CO2_per_kg = \
        GH2_TML_stor_compr_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert storage compressor energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_TML_stor_compr_ghg_g_CO2_per_MJ = \
        GH2_TML_stor_compr_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        GH2_TML_stor_compr_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        GH2_TML_stor_compr_ghg_g_CO2_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # storage *compressor* energy cost
    
    # calculate storage compressor energy cost ($/kg H2)
    GH2_TML_stor_compr_elec_cost_usd_per_kg = \
        GH2_TML_stor_compr_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate storage compressor energy cost ($/yr)
    GH2_TML_stor_compr_elec_cost_usd_per_yr = \
        GH2_TML_stor_compr_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        GH2_TML_stor_compr_elec_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        GH2_TML_stor_compr_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # storage *compressor* installed cost and annual O&M cost

    # calculate storage compressor installed cost ($) and annual O&M 
    # cost ($/yr), both in output dollar year 
    GH2_TML_stor_compr_inst_cost_usd, \
    GH2_TML_stor_compr_om_cost_usd_per_yr, \
    GH2_TML_stor_compr_dollar_year = \
        compressor_fixed_costs(
            compr_power_kW_per_stg = GH2_TML_stor_compr_power_kW_per_stg, 
            num_stgs = GH2_TML_stor_compr_num_stgs, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate storage compressor O&M cost ($/kg H2)
    GH2_TML_stor_compr_om_cost_usd_per_kg = \
        GH2_TML_stor_compr_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        GH2_TML_stor_compr_om_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        GH2_TML_stor_compr_om_cost_usd_per_kg
        ])
            
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # terminal storage installed cost and annual O&M cost

    # calculate total storage capacity required (kg) at compressed hydrogen 
    # terminal
    GH2_TML_stor_tot_capacity_kg, _ = \
        GH2_terminal_storage_size(
            H2_flow_kg_per_day = TML_H2_flow_kg_per_day,
            stor_amt_days = GH2_TML_stor_amt_days
            )
    
    # calculate storage installed cost ($) and annual O&M cost ($/yr), 
    # both in output dollar year
    # TODO: revisit $ and $/yr when adding losses ("gross" vs. delivered)
    GH2_TML_stor_inst_cost_usd, \
    GH2_TML_stor_om_cost_usd_per_yr, \
    GH2_TML_stor_dollar_year = \
        GH2_terminal_storage_fixed_costs(
            stor_tot_capacity_kg = GH2_TML_stor_tot_capacity_kg, 
            output_dollar_year = output_dollar_year
            )
                
    # calculate storage O&M cost ($/kg H2)
    GH2_TML_stor_om_cost_usd_per_kg = \
        GH2_TML_stor_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'compressed gas storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        GH2_TML_stor_om_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'compressed gas storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        GH2_TML_stor_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # terminal total capital investment and "other" annual O&M costs
    
    # "Other" O&M costs include insurance, property taxes, licensing and 
    # permits. Operation, maintenance, and repair costs for individual 
    # terminal components (e.g., compressor) are calculated separately.
    
    # calculate terminal total initial capital investment ($)
    # (= loading compressor, storage compressor, and storage tank 
    # installed costs)
    GH2_TML_init_cap_inv_usd = \
        GH2_TML_load_compr_inst_cost_usd + \
        GH2_TML_stor_compr_inst_cost_usd + \
        GH2_TML_stor_inst_cost_usd
        
    # calculate terminal cost allocations (%) to loading compressor, 
    # storage compressor, and storage tanks
    # % of terminal total initial capital investment
    # use to allocate total capital investment, other O&M costs, and labor 
    # cost
    GH2_TML_load_compr_cost_perc = \
        GH2_TML_load_compr_inst_cost_usd / \
        GH2_TML_init_cap_inv_usd
    GH2_TML_stor_compr_cost_perc = \
        GH2_TML_stor_compr_inst_cost_usd / \
        GH2_TML_init_cap_inv_usd
    GH2_TML_stor_cost_perc = \
        GH2_TML_stor_inst_cost_usd / \
        GH2_TML_init_cap_inv_usd
        
    # check whether cost allocations (%) sum to one
    # raise error if false
    if abs(
            GH2_TML_load_compr_cost_perc + \
            GH2_TML_stor_compr_cost_perc + \
            GH2_TML_stor_cost_perc - \
            1.0
            ) >= 1.0e-9:
        raise ValueError(
            'Component cost allocations need to sum to one.'
            )
        
    # check if all terminal components have the same dollar year
    # if true, assign dollar year of refueling station costs to the dollar 
    # year of one of the components 
    if (GH2_TML_load_compr_dollar_year == \
        GH2_TML_stor_compr_dollar_year) \
        and (GH2_TML_stor_compr_dollar_year == \
             GH2_TML_stor_dollar_year):
        GH2_TML_dollar_year = GH2_TML_load_compr_dollar_year
    else:
        raise ValueError(
            'Dollar year of components need to match.'
            )

    # calculate terminal total capital investment ($, output dollar year) 
    GH2_TML_tot_cap_inv_usd, \
    GH2_TML_cap_cost_dollar_year = \
        non_station_total_capital_investment(
            init_cap_inv_usd = GH2_TML_init_cap_inv_usd, 
            input_dollar_year = GH2_TML_dollar_year
            )
    
    # calculate terminal "other" annual O&M costs ($/yr, output dollar year)
    GH2_TML_om_cost_usd_per_yr, \
    GH2_TML_om_cost_dollar_year = \
        other_om_cost(
            tot_cap_inv_usd = GH2_TML_tot_cap_inv_usd,
            input_dollar_year = GH2_TML_cap_cost_dollar_year
            )
    
    # calculate terminal "other" O&M costs ($/kg H2)
    GH2_TML_om_cost_usd_per_kg = \
        GH2_TML_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign "other" O&M costs to loading compressor, storage compressor, 
    # and storage tanks
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        GH2_TML_om_cost_usd_per_yr * \
            GH2_TML_load_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        GH2_TML_om_cost_usd_per_kg * \
            GH2_TML_load_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        GH2_TML_om_cost_usd_per_yr * \
            GH2_TML_stor_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        GH2_TML_om_cost_usd_per_kg * \
            GH2_TML_stor_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'compressed gas storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        GH2_TML_om_cost_usd_per_yr * \
            GH2_TML_stor_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'compressed gas storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        GH2_TML_om_cost_usd_per_kg * \
            GH2_TML_stor_cost_perc
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: terminal annual labor cost 

    # calculate terminal annual labor cost ($/yr, output dollar year), 
    # including overhead and G&A
    GH2_TML_labor_cost_usd_per_yr, \
    GH2_TML_labor_cost_dollar_year = \
        non_station_labor_cost(
            H2_flow_kg_per_day = TML_H2_flow_kg_per_day, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate terminal labor cost ($/kg H2)
    GH2_TML_labor_cost_usd_per_kg = \
        GH2_TML_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign labor cost to loading compressor, storage compressor, and 
    # storage tanks
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        GH2_TML_labor_cost_usd_per_yr * \
            GH2_TML_load_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        GH2_TML_labor_cost_usd_per_kg * \
            GH2_TML_load_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        GH2_TML_labor_cost_usd_per_yr * \
            GH2_TML_stor_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        GH2_TML_labor_cost_usd_per_kg * \
            GH2_TML_stor_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'compressed gas storage', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        GH2_TML_labor_cost_usd_per_yr * \
            GH2_TML_stor_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'compressed gas storage', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        GH2_TML_labor_cost_usd_per_kg * \
            GH2_TML_stor_cost_perc
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # loading compressor levelized capital cost
    
    # calculate loading compressor total capital investment ($) 
    # (= terminal total capital investment allocated to loading compressor)
    GH2_TML_load_compr_tot_cap_inv_usd = \
        GH2_TML_load_compr_cost_perc * GH2_TML_tot_cap_inv_usd
    
    # calculate loading compressor levelized capital cost 
    # ($/yr, output dollar year)
    GH2_TML_load_compr_lev_cap_cost_usd_per_yr, \
    GH2_TML_load_compr_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = GH2_TML_load_compr_tot_cap_inv_usd, 
            life_yr = TML_compr_life_yr, 
            depr_yr = TML_compr_depr_yr,
            input_dollar_year = GH2_TML_cap_cost_dollar_year
            )
    
    # calculate loading compressor levelized capital cost ($/kg H2)
    GH2_TML_load_compr_lev_cap_cost_usd_per_kg = \
        GH2_TML_load_compr_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
            
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        GH2_TML_load_compr_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'loading compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        GH2_TML_load_compr_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # storage compressor levelized capital cost
    
    # calculate storage compressor total capital investment ($) 
    # (= terminal total capital investment allocated to storage compressor)
    GH2_TML_stor_compr_tot_cap_inv_usd = \
        GH2_TML_stor_compr_cost_perc * GH2_TML_tot_cap_inv_usd
    
    # calculate storage compressor levelized capital cost 
    # ($/yr, output dollar year)
    GH2_TML_stor_compr_lev_cap_cost_usd_per_yr, \
    GH2_TML_stor_compr_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = GH2_TML_stor_compr_tot_cap_inv_usd, 
            life_yr = TML_compr_life_yr, 
            depr_yr = TML_compr_depr_yr,
            input_dollar_year = GH2_TML_cap_cost_dollar_year
            )
    
    # calculate storage compressor levelized capital cost ($/kg H2)
    GH2_TML_stor_compr_lev_cap_cost_usd_per_kg = \
        GH2_TML_stor_compr_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        GH2_TML_stor_compr_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'compression', 
        'storage compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        GH2_TML_stor_compr_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - compressed hydrogen: 
    # storage levelized capital cost 

    # calculate storage total capital investment ($) 
    # (= terminal total capital investment allocated to storage)
    GH2_TML_stor_tot_cap_inv_usd = \
        GH2_TML_stor_cost_perc * GH2_TML_tot_cap_inv_usd
    
    # calculate storage levelized capital cost 
    # ($/yr, output dollar year)
    GH2_TML_stor_lev_cap_cost_usd_per_yr, \
    GH2_TML_stor_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = GH2_TML_stor_tot_cap_inv_usd, 
            life_yr = TML_stor_life_yr, 
            depr_yr = TML_stor_depr_yr,
            input_dollar_year = GH2_TML_cap_cost_dollar_year
            )
    
    # calculate storage levelized capital cost ($/kg H2)
    GH2_TML_stor_lev_cap_cost_usd_per_kg = \
        GH2_TML_stor_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'compressed gas storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        GH2_TML_stor_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'compressed gas storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        GH2_TML_stor_lev_cap_cost_usd_per_kg
        ])
        
    #%% CALCULATIONS: COMPRESSED GASEOUS HYDROGEN DELIVERY ("GH2")
    # TRANSPORT @ TRUCK

    # ------------------------------------------------------------------------
    # transport - compressed hydrogen: 
    # truck fuel consumption, number of trucks required, number of deliveries 
    # per day, total trip time
    
    # calculate truck fuel consumption (gallon/kg H2), number of trucks, 
    # number of deliveries per day, total trip time (hr/trip)
    # TODO: revisit gal/kg H2 when adding losses (transported vs. delivered)
    GH2_truck_fuel_gal_per_kg, \
    GH2_num_trucks, \
    GH2_truck_num_delivs_per_day, \
    GH2_truck_trip_time_hr = \
        transport_energy(
            deliv_dist_mi = deliv_dist_mi_rt, 
            speed_mi_per_hr = truck_speed_mi_per_hr, 
            load_unload_time_hr = gas_truck_load_unload_time_hr,
            fuel_econ_mi_per_gal = truck_fuel_econ_mi_per_gal,
            deliv_capacity_kg = gas_truck_deliv_capacity_kgH2, 
            cargo_flow_kg_per_day = TML_H2_flow_kg_per_day
            )
    
    # convert truck fuel consumption to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_truck_fuel_MJ_per_MJ = \
        GH2_truck_fuel_gal_per_kg * low_heat_val_diesel_MJ_per_gal / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy consumption', 
        'diesel consumption', 
        'gallon/kg H2', 
        GH2_truck_fuel_gal_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy consumption', 
        'diesel consumption', 
        'MJ/MJ H2 (LHV)', 
        GH2_truck_fuel_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # transport - compressed hydrogen: 
    # truck fuel emissions

    # calculate truck fuel emissions (kg CO2/kg H2)
    GH2_truck_ghg_kg_CO2_per_kg = \
        GH2_truck_fuel_gal_per_kg * diesel_ghg_kg_CO2_per_gal
    
    # convert truck fuel emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_truck_ghg_g_CO2_per_MJ = \
        GH2_truck_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        GH2_truck_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        GH2_truck_ghg_g_CO2_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # transport - compressed hydrogen: truck fuel cost
    
    # calculate truck fuel cost ($/kg H2)
    GH2_truck_fuel_cost_usd_per_kg = \
        GH2_truck_fuel_gal_per_kg * diesel_cost_usd_per_gal
    
    # calculate truck fuel cost ($/yr)
    GH2_truck_fuel_cost_usd_per_yr = \
        GH2_truck_fuel_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy cost', 
        'fuel cost', 
        '$/yr', 
        GH2_truck_fuel_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy cost', 
        'fuel cost', 
        '$/kg H2', 
        GH2_truck_fuel_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # transport - compressed hydrogen: truck capital cost and annual O&M cost
    
    # calculate compressed gaseous hydrogen truck capital cost 
    # ($, output dollar year) 
    GH2_truck_cap_cost_usd, \
    GH2_truck_cap_cost_dollar_year = \
        gas_truck_capital_cost(
            num_trucks = GH2_num_trucks,
            output_dollar_year = output_dollar_year
            )
    
    # calculate number of deliveries (truck-trips) per year for compressed 
    # gaseous hydrogen
    GH2_truck_num_delivs_per_yr = \
        GH2_truck_num_delivs_per_day * day_per_yr 
    
    # calculate truck annual O&M cost ($/yr, output dollar year)
    GH2_truck_om_cost_usd_per_yr, \
    GH2_truck_om_cost_dollar_year = \
        truck_om_cost(
            deliv_dist_mi = deliv_dist_mi_rt, 
            num_delivs_per_yr = GH2_truck_num_delivs_per_yr, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate truck O&M cost ($/kg H2)
    GH2_truck_om_cost_usd_per_kg = \
        GH2_truck_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        GH2_truck_om_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        GH2_truck_om_cost_usd_per_kg
       ])
    
    # ------------------------------------------------------------------------
    # transport - compressed hydrogen: truck annual labor cost
    
    # calculate truck annual labor cost ($/yr, output dollar year), 
    # including overhead and G&A
    GH2_truck_labor_cost_usd_per_yr, \
    GH2_truck_labor_cost_dollar_year = \
        truck_labor_cost(
            num_delivs_per_yr = GH2_truck_num_delivs_per_yr, 
            trip_time_hr = GH2_truck_trip_time_hr, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate truck labor cost ($/kg H2)
    GH2_truck_labor_cost_usd_per_kg = \
        GH2_truck_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        GH2_truck_labor_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        GH2_truck_labor_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # transport - compressed hydrogen: truck levelized capital cost
    
    # calculate truck levelized capital cost ($/yr, output dollar year)
    # (truck total capital investment = truck capital cost)
    GH2_truck_lev_cap_cost_usd_per_yr, \
    GH2_truck_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = GH2_truck_cap_cost_usd, 
            life_yr = truck_life_yr, 
            depr_yr = truck_depr_yr,
            input_dollar_year = GH2_truck_cap_cost_dollar_year
            )
    
    # calculate truck levelized capital cost ($/kg H2)
    GH2_truck_lev_cap_cost_usd_per_kg = \
        GH2_truck_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        GH2_truck_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        GH2_truck_lev_cap_cost_usd_per_kg
        ])
        
    #%% CALCULATIONS: COMPRESSED GASEOUS HYDROGEN DELIVERY ("GH2")
    # RECONDITIONING @ REFUELING STATION

    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station compressor energy consumption and size
    
    # define compressor inlet pressure (bar) for *power* calculations
    GH2_STN_in_pres_bar = gas_truck_min_tube_pres_atm
    
    # calculate compressor average inlet pressure (bar) 
    # for *energy* calculations
    # log-mean of minimum and maximum tube trailer operating pressures
    GH2_STN_avg_in_pres_bar = \
        (gas_truck_max_tube_pres_atm - gas_truck_min_tube_pres_atm) * \
        Pa_per_atm / Pa_per_bar / \
        math.log(gas_truck_max_tube_pres_atm / gas_truck_min_tube_pres_atm)
    
    # calculate compressor power (kW) and size (kW/stage)
    GH2_STN_compr_tot_power_kW, \
    GH2_STN_compr_power_kW_per_stg, \
    GH2_STN_compr_num_stgs = \
        compressor_power_and_size(
            out_pres_bar = GH2_STN_out_pres_bar, 
            in_pres_bar = GH2_STN_in_pres_bar, 
            in_temp_K = STN_H2_temp_K, 
            gas_flow_mol_per_sec = STN_H2_flow_mol_per_sec, 
            compressibility = 1.28
            )
    
    # calculate compressor average power (kW) 
    # for *energy* calculations
    GH2_STN_compr_avg_power_kW, _, _ = \
        compressor_power_and_size(
            out_pres_bar = GH2_STN_out_pres_bar, 
            in_pres_bar = GH2_STN_avg_in_pres_bar, 
            in_temp_K = STN_H2_temp_K, 
            gas_flow_mol_per_sec = STN_H2_flow_mol_per_sec, 
            compressibility = 1.28
            )
    
    # calculate refueling station compressor energy (kWh/kg H2)
    GH2_STN_compr_elec_kWh_per_kg = \
        GH2_STN_compr_avg_power_kW * target_num_stns / \
        tot_H2_deliv_kg_per_hr
    
    # convert compressor energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_STN_compr_elec_MJ_per_MJ = \
        GH2_STN_compr_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        GH2_STN_compr_elec_kWh_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        GH2_STN_compr_elec_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station compressor energy emissions

    # calculate refueling station compressor energy emissions (kg CO2/kg H2)
    GH2_STN_compr_ghg_kg_CO2_per_kg = \
        GH2_STN_compr_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert compressor energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_STN_compr_ghg_g_CO2_per_MJ = \
        GH2_STN_compr_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        GH2_STN_compr_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        GH2_STN_compr_ghg_g_CO2_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station compressor energy cost
    
    # calculate refueling station compressor energy cost ($/kg H2)
    GH2_STN_compr_elec_cost_usd_per_kg = \
        GH2_STN_compr_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate refueling station compressor energy cost ($/yr)
    GH2_STN_compr_elec_cost_usd_per_yr = \
        GH2_STN_compr_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        GH2_STN_compr_elec_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        GH2_STN_compr_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station compressor installed cost and annual O&M cost
    
    # calculate refueling station compressor installed cost ($) and annual O&M 
    # cost ($/yr) per station, both in output dollar year
    GH2_STN_compr_inst_cost_usd_per_stn, \
    GH2_STN_compr_om_cost_usd_per_yr_per_stn, \
    GH2_STN_compr_dollar_year = \
        compressor_fixed_costs(
            compr_power_kW_per_stg = GH2_STN_compr_power_kW_per_stg, 
            num_stgs = GH2_STN_compr_num_stgs, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate refueling station compressor installed cost ($)
    # sum of all stations
    GH2_STN_compr_inst_cost_usd = \
        GH2_STN_compr_inst_cost_usd_per_stn * target_num_stns

    # calculate refueling station compressor O&M cost ($/yr)
    # sum of all stations
    GH2_STN_compr_om_cost_usd_per_yr = \
        GH2_STN_compr_om_cost_usd_per_yr_per_stn * target_num_stns 

    # calculate refueling station compressor O&M cost ($/kg H2)
    GH2_STN_compr_om_cost_usd_per_kg = \
        GH2_STN_compr_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        GH2_STN_compr_om_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        GH2_STN_compr_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station refrigerator energy consumption
    
    # calculate refueling station refrigerator energy (kWh/kg H2)
    GH2_STN_refrig_elec_kWh_per_kg = \
        heat_exchanger_energy(
            out_temp_K = STN_dispens_temp_K,
            in_temp_K = STN_H2_temp_K
            )
    
    # convert refrigerator energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_STN_refrig_elec_MJ_per_MJ = \
        GH2_STN_refrig_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        GH2_STN_refrig_elec_kWh_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        GH2_STN_refrig_elec_MJ_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station refrigerator energy emissions

    # calculate refueling station refrigerator energy emissions (kg CO2/kg H2)
    GH2_STN_refrig_ghg_kg_CO2_per_kg = \
        GH2_STN_refrig_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert refrigerator energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    GH2_STN_refrig_ghg_g_CO2_per_MJ = \
        GH2_STN_refrig_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        GH2_STN_refrig_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        GH2_STN_refrig_ghg_g_CO2_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station refrigerator energy cost
    
    # calculate refueling station refrigerator energy cost ($/kg H2)
    GH2_STN_refrig_elec_cost_usd_per_kg = \
        GH2_STN_refrig_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate refueling station refrigerator energy cost ($/yr)
    GH2_STN_refrig_elec_cost_usd_per_yr = \
        GH2_STN_refrig_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        GH2_STN_refrig_elec_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        GH2_STN_refrig_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station refrigerator installed cost and 
    # annual O&M cost
    
    # calculate number of refrigerators needed at refueling station 
    # (= number of hoses)
    # assume linear relative to station capacity 
    # (HDSAM V3.1: 1000 kg H2/day --> 4 hoses, 4 refrigerators)
    GH2_STN_num_refrigs = \
        target_stn_capacity_kg_per_day / (1000.0 / 4)
    
    # calculate refueling station refrigerator installed cost ($) and annual 
    # O&M cost ($/yr) per station, both in output dollar year
    GH2_STN_refrig_inst_cost_usd_per_stn, \
    GH2_STN_refrig_om_cost_usd_per_yr_per_stn, \
    GH2_STN_refrig_dollar_year = \
        heat_exchanger_fixed_costs(
            out_temp_K = STN_dispens_temp_K, 
            num_hx = GH2_STN_num_refrigs, 
            output_dollar_year = output_dollar_year
            )    
    
    # calculate refueling station refrigerator installed cost ($) 
    # sum of all stations
    GH2_STN_refrig_inst_cost_usd = \
        GH2_STN_refrig_inst_cost_usd_per_stn * target_num_stns

    # calculate refueling station refrigerator O&M cost ($/yr)
    # sum of all stations
    GH2_STN_refrig_om_cost_usd_per_yr = \
        GH2_STN_refrig_om_cost_usd_per_yr_per_stn * target_num_stns 

    # calculate refueling station refrigerator O&M cost ($/kg H2)
    GH2_STN_refrig_om_cost_usd_per_kg = \
        GH2_STN_refrig_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        GH2_STN_refrig_om_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        GH2_STN_refrig_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station cascade storage installed cost and annual O&M cost

    # calculate total cascade storage capacity required (kg) at 
    # refueling station
    GH2_STN_casc_stor_tot_capacity_kg = \
        station_cascade_storage_size(
            stn_capacity_kg_per_day = target_stn_capacity_kg_per_day,
            casc_stor_size_frac = GH2_STN_casc_stor_size_frac
            )
    
    # calculate cascade storage installed cost ($) and annual O&M 
    # cost ($/yr) per station, both in output dollar year
    GH2_STN_casc_stor_inst_cost_usd_per_stn, \
    GH2_STN_casc_stor_om_cost_usd_per_yr_per_stn, \
    GH2_STN_casc_stor_dollar_year = \
        station_cascade_storage_fixed_costs(
            stor_tot_capacity_kg = GH2_STN_casc_stor_tot_capacity_kg, 
            output_dollar_year = output_dollar_year
            )
                
    # calculate cascade storage installed cost ($)
    # sum of all stations
    GH2_STN_casc_stor_inst_cost_usd = \
        GH2_STN_casc_stor_inst_cost_usd_per_stn * target_num_stns

    # calculate cascade storage O&M cost ($/yr)
    # sum of all stations
    GH2_STN_casc_stor_om_cost_usd_per_yr = \
        GH2_STN_casc_stor_om_cost_usd_per_yr_per_stn * target_num_stns

    # calculate cascade storage O&M cost ($/kg H2)
    GH2_STN_casc_stor_om_cost_usd_per_kg = \
        GH2_STN_casc_stor_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        GH2_STN_casc_stor_om_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        GH2_STN_casc_stor_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station total capital investment and "other" annual O&M costs
        
    # "Other" O&M costs include insurance, property taxes, licensing and 
    # permits. Operation, maintenance, and repair costs for individual 
    # refueling station components (e.g., compressor) are calculated 
    # separately.
    
    # calculate refueling station total initial capital investment ($)
    # (= compressor + refrigerator + cascade storage for *compressed* 
    # hydrogen refueling station)
    # sum of all stations
    GH2_STN_init_cap_inv_usd = \
        GH2_STN_compr_inst_cost_usd + \
        GH2_STN_refrig_inst_cost_usd + \
        GH2_STN_casc_stor_inst_cost_usd
    
    # calculate refueling station cost allocations (%) to compressor, 
    # refrigerator, and cascade storage
    # % of refueling station total initial capital investment
    # use to allocate total capital investment, other O&M costs, and labor 
    # cost
    GH2_STN_compr_cost_perc = \
        GH2_STN_compr_inst_cost_usd / \
        GH2_STN_init_cap_inv_usd
    GH2_STN_refrig_cost_perc = \
        GH2_STN_refrig_inst_cost_usd / \
        GH2_STN_init_cap_inv_usd
    GH2_STN_casc_stor_cost_perc = \
        GH2_STN_casc_stor_inst_cost_usd / \
        GH2_STN_init_cap_inv_usd
    
    # check whether cost allocations (%) sum to one
    # raise error if false
    if abs(
            GH2_STN_compr_cost_perc + \
            GH2_STN_refrig_cost_perc + \
            GH2_STN_casc_stor_cost_perc - \
            1.0
            ) >= 1.0e-9:
        raise ValueError(
            'Component cost allocations need to sum to one.'
            )
        
    # check if all refueling station components have the same dollar year
    # if true, assign dollar year of refueling station costs to the dollar 
    # year of one of the components 
    if (GH2_STN_compr_dollar_year == \
        GH2_STN_refrig_dollar_year) \
        and (GH2_STN_refrig_dollar_year == \
             GH2_STN_casc_stor_dollar_year):
        GH2_STN_dollar_year = GH2_STN_compr_dollar_year
    else:
        raise ValueError(
            'Dollar year of components need to match.'
            )
    
    # calculate refueling station total capital investment 
    # ($, output dollar year), sum of all stations
    GH2_STN_tot_cap_inv_usd, \
    GH2_STN_cap_cost_dollar_year = \
        station_total_capital_investment(
            init_cap_inv_usd = GH2_STN_init_cap_inv_usd, 
            input_dollar_year = GH2_STN_dollar_year
            )
    
    # calculate refueling station "other" annual O&M costs 
    # ($/yr, output dollar year), sum of all stations
    GH2_STN_om_cost_usd_per_yr, \
    GH2_STN_om_cost_dollar_year = \
        other_om_cost(
            tot_cap_inv_usd = GH2_STN_tot_cap_inv_usd,
            input_dollar_year = GH2_STN_cap_cost_dollar_year
            )
    
    # calculate refueling station "other" O&M costs ($/kg H2)
    GH2_STN_om_cost_usd_per_kg = \
        GH2_STN_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign "other" O&M costs to compressor, refrigerator, and 
    # cascade storage
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        GH2_STN_om_cost_usd_per_yr * \
            GH2_STN_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        GH2_STN_om_cost_usd_per_kg * \
            GH2_STN_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        GH2_STN_om_cost_usd_per_yr * \
            GH2_STN_refrig_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        GH2_STN_om_cost_usd_per_kg * \
            GH2_STN_refrig_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        GH2_STN_om_cost_usd_per_yr * \
            GH2_STN_casc_stor_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        GH2_STN_om_cost_usd_per_kg * \
            GH2_STN_casc_stor_cost_perc
        ])
        
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station annual labor cost
    
    # calculate refueling station annual labor cost 
    # ($/yr, output dollar year) per station, including overhead and G&A
    GH2_STN_labor_cost_usd_per_yr_per_stn, \
    GH2_STN_labor_cost_dollar_year = \
        station_labor_cost(
            H2_flow_kg_per_day = target_stn_capacity_kg_per_day, 
            output_dollar_year = output_dollar_year
            ) 
    
    # calculate refueling station labor cost ($/yr)
    # sum of all stations
    GH2_STN_labor_cost_usd_per_yr = \
        GH2_STN_labor_cost_usd_per_yr_per_stn * target_num_stns

    # calculate refueling station labor cost ($/kg H2)
    GH2_STN_labor_cost_usd_per_kg = \
        GH2_STN_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign labor cost to compressor, refrigerator, and cascade storage
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        GH2_STN_labor_cost_usd_per_yr * \
            GH2_STN_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        GH2_STN_labor_cost_usd_per_kg * \
            GH2_STN_compr_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        GH2_STN_labor_cost_usd_per_yr * \
            GH2_STN_refrig_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        GH2_STN_labor_cost_usd_per_kg * \
            GH2_STN_refrig_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        GH2_STN_labor_cost_usd_per_yr * \
            GH2_STN_casc_stor_cost_perc
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        GH2_STN_labor_cost_usd_per_kg * \
            GH2_STN_casc_stor_cost_perc
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station compressor levelized capital cost
    
    # calculate refueling station compressor total capital investment ($) 
    # (= refueling station total capital investment allocated to compressor)
    # sum of all stations
    GH2_STN_compr_tot_cap_inv_usd = \
        GH2_STN_compr_cost_perc * GH2_STN_tot_cap_inv_usd
    
    # calculate refueling station compressor levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    GH2_STN_compr_lev_cap_cost_usd_per_yr, \
    GH2_STN_compr_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = GH2_STN_compr_tot_cap_inv_usd, 
            life_yr = STN_compr_life_yr, 
            depr_yr = STN_compr_depr_yr,
            input_dollar_year = GH2_STN_dollar_year
            )
    
    # calculate refueling station compressor levelized capital cost ($/kg H2)
    GH2_STN_compr_lev_cap_cost_usd_per_kg = \
        GH2_STN_compr_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        GH2_STN_compr_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        GH2_STN_compr_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station refrigerator levelized capital cost
        
    # calculate refueling station refrigerator total capital investment ($) 
    # (= refueling station total capital investment allocated to refrigerator)
    # sum of all stations
    GH2_STN_refrig_tot_cap_inv_usd = \
        GH2_STN_refrig_cost_perc * GH2_STN_tot_cap_inv_usd
    
    # calculate refueling station refrigerator levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    GH2_STN_refrig_lev_cap_cost_usd_per_yr, \
    GH2_STN_refrig_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = GH2_STN_refrig_tot_cap_inv_usd, 
            life_yr = STN_refrig_life_yr, 
            depr_yr = STN_refrig_depr_yr,
            input_dollar_year = GH2_STN_dollar_year
            )
    
    # calculate refueling station refrigerator levelized capital cost 
    # ($/kg H2)
    GH2_STN_refrig_lev_cap_cost_usd_per_kg = \
        GH2_STN_refrig_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        GH2_STN_refrig_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        GH2_STN_refrig_lev_cap_cost_usd_per_kg
        ])
        
    # ------------------------------------------------------------------------
    # reconditioning - compressed hydrogen: 
    # refueling station cascade storage levelized capital cost
        
    # calculate refueling station cascade storage total capital investment ($) 
    # (= refueling station total capital investment allocated to cascade 
    # storage)
    # sum of all stations
    GH2_STN_casc_stor_tot_cap_inv_usd = \
        GH2_STN_casc_stor_cost_perc * GH2_STN_tot_cap_inv_usd
    
    # calculate refueling station cascade storage levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    GH2_STN_casc_stor_lev_cap_cost_usd_per_yr, \
    GH2_STN_casc_stor_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = GH2_STN_casc_stor_tot_cap_inv_usd, 
            life_yr = STN_stor_life_yr, 
            depr_yr = STN_stor_depr_yr,
            input_dollar_year = GH2_STN_dollar_year
            )
    
    # calculate refueling station cascade storage levelized capital cost 
    # ($/kg H2)
    GH2_STN_casc_stor_lev_cap_cost_usd_per_kg = \
        GH2_STN_casc_stor_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        GH2_STN_casc_stor_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'compressed hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        GH2_STN_casc_stor_lev_cap_cost_usd_per_kg
        ])
        
    #%% CALCULATIONS: LIQUID HYDROGEN DELIVERY ("LH2")
    # PRECONDITIONING @ LIQUEFIER

    # ------------------------------------------------------------------------
    # production - liquid hydrogen: hydrogen purchase cost
    
    # calculate hydrogen purchase cost ($/yr)   
    LH2_TML_H2_purc_cost_usd_per_yr = \
        purc_cost_H2_usd_per_kg * LH2_TML_H2_in_flow_kg_per_day * \
        day_per_yr
        
    # calculate hydrogen purchase cost ($/kg H2)   
    LH2_TML_H2_purc_cost_usd_per_kg = \
        LH2_TML_H2_purc_cost_usd_per_yr / tot_H2_deliv_kg_per_yr

    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'O&M cost', 
        'purchase cost', 
        '$/yr', 
        LH2_TML_H2_purc_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'O&M cost', 
        'purchase cost', 
        '$/kg H2', 
        LH2_TML_H2_purc_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # production - liquid hydrogen: emissions of purchased hydrogen

    # calculate emissions of purchased hydrogen (kg CO2-eq/kg H2)
    LH2_TML_H2_purc_ghg_kg_CO2_per_kg = \
        H2_prod_ghg_kg_CO2_per_kg * LH2_TML_H2_in_flow_kg_per_day * \
        day_per_yr / tot_H2_deliv_kg_per_yr
    
    # convert emissions of purchased hydrogen to g CO2-eq/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_TML_H2_purc_ghg_g_CO2_per_MJ = \
        LH2_TML_H2_purc_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'emissions', 
        'upstream emissions', 
        'kg CO2/kg H2', 
        LH2_TML_H2_purc_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'emissions', 
        'upstream emissions', 
        'g CO2/MJ H2 (LHV)', 
        LH2_TML_H2_purc_ghg_g_CO2_per_MJ
        ])

    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: liquefier energy consumption
    
    # calculate liquefier energy (kWh/kg H2) for assumed liquefier size 
    # (tonne H2/day)
    # TODO: revisit kWh/kg when adding number of liquefiers
    LH2_liquef_elec_kWh_per_kg = \
        liquefier_energy(
            liquef_size_tonne_per_day = liquef_size_tonne_per_day
            )
    
    # convert liquefier energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_liquef_elec_MJ_per_MJ = \
        LH2_liquef_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        LH2_liquef_elec_kWh_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        LH2_liquef_elec_MJ_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: liquefier energy emissions

    # calculate liquefier energy emissions (kg CO2/kg H2)
    LH2_liquef_ghg_kg_CO2_per_kg = \
        LH2_liquef_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert liquefier energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_liquef_ghg_g_CO2_per_MJ = \
        LH2_liquef_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        LH2_liquef_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        LH2_liquef_ghg_g_CO2_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: liquefier energy cost
    
    # calculate liquefier energy cost ($/kg H2)
    LH2_liquef_elec_cost_usd_per_kg = \
        LH2_liquef_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate liquefier energy cost ($/yr)
    LH2_liquef_elec_cost_usd_per_yr = \
        LH2_liquef_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        LH2_liquef_elec_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        LH2_liquef_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # liquefier installed cost and annual O&M cost
    
    # calculate liquefier installed cost ($) and annual O&M cost ($/yr), 
    # both in output dollar year
    # TODO: revisit kWh/kg when adding number of liquefiers
    LH2_liquef_inst_cost_usd, \
    LH2_liquef_om_cost_usd_per_yr, \
    LH2_liquef_dollar_year = \
        liquefier_fixed_costs(
            liquef_size_tonne_per_day = liquef_size_tonne_per_day,
            output_dollar_year = output_dollar_year
            )
    
    # calculate liquefier O&M cost ($/kg H2)
    LH2_liquef_om_cost_usd_per_kg = \
        LH2_liquef_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        LH2_liquef_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        LH2_liquef_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # liquefier total capital investment and "other" annual O&M costs
        
    # "Other" O&M costs include insurance, property taxes, licensing and 
    # permits. Operation, maintenance, and repair costs for liquefier are 
    # calculated separately.
    
    # calculate liquefier total initial capital investment ($)
    # (= liquefier installed cost)
    LH2_liquef_init_cap_inv_usd = LH2_liquef_inst_cost_usd
    
    # calculate liquefier total capital investment ($, output dollar year)
    # NOTE: only owner's cost applies to liquefier (= 12% in HDSAM V3.1)
    LH2_liquef_tot_cap_inv_usd, \
    LH2_liquef_cap_cost_dollar_year = \
        non_station_total_capital_investment(
            init_cap_inv_usd = LH2_liquef_init_cap_inv_usd, 
            input_dollar_year = LH2_liquef_dollar_year, 
            indir_cost_perc_override = 0.12
            )
    
    # calculate liquefier "other" annual O&M costs ($/yr, output dollar year)
    LH2_liquef_om_cost_usd_per_yr, \
    LH2_liquef_om_cost_dollar_year = \
        other_om_cost(
            tot_cap_inv_usd = LH2_liquef_tot_cap_inv_usd,
            input_dollar_year = LH2_liquef_cap_cost_dollar_year
            )
    
    # calculate liquefier "other" O&M costs ($/kg H2)
    LH2_liquef_om_cost_usd_per_kg = \
        LH2_liquef_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        LH2_liquef_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        LH2_liquef_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: liquefier annual labor cost

    # calculate liquefier annual labor cost ($/yr, output dollar year), 
    # including overhead and G&A
    LH2_liquef_labor_cost_usd_per_yr, \
    LH2_liquef_labor_cost_dollar_year = \
        non_station_labor_cost(
            H2_flow_kg_per_day = LH2_TML_H2_in_flow_kg_per_day, 
            output_dollar_year = output_dollar_year
            ) 
    
    # calculate terminal labor cost ($/kg H2)
    LH2_liquef_labor_cost_usd_per_kg = \
        LH2_liquef_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign labor cost to liquefier
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        LH2_liquef_labor_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        LH2_liquef_labor_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: liquefier levelized capital cost
    
    # calculate liquefier levelized capital cost ($/yr, output dollar year)
    LH2_liquef_lev_cap_cost_usd_per_yr, \
    LH2_liquef_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = LH2_liquef_tot_cap_inv_usd, 
            life_yr = liquef_life_yr, 
            depr_yr = liquef_depr_yr,
            input_dollar_year = LH2_liquef_cap_cost_dollar_year
            )
    
    # calculate liquefier levelized capital cost ($/kg H2)
    LH2_liquef_lev_cap_cost_usd_per_kg = \
        LH2_liquef_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        LH2_liquef_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'liquefier', 
        'liquefaction', 
        'liquefier', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        LH2_liquef_lev_cap_cost_usd_per_kg
        ])
    
    #%% CALCULATIONS: LIQUID HYDROGEN DELIVERY ("LH2")
    # PRECONDITIONING @ TERMINAL

    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # low-head truck loading pump energy consumption and size
    
    # TODO: revisit flowrate through pump - incoroporate fill time, peak 
    # demand, outage, etc.; for now, use hydrogen flowrate through terminal
        
    # calculate loading pump inlet pressure (bar)
    LH2_TML_in_pres_bar = \
        LH2_TML_in_pres_atm * Pa_per_atm / Pa_per_bar
    
    # calculate loading pump outlet pressure (bar)
    LH2_TML_out_pres_bar = \
        LH2_TML_out_pres_atm * Pa_per_atm / Pa_per_bar

    # calculate loading pump power (kW) and size (kW/pump) 
    # at refueling station
    LH2_TML_pump_tot_power_kW, \
    LH2_TML_pump_power_kW_per_pump, \
    LH2_TML_num_pumps = \
        pump_power_and_size(
            out_pres_bar = LH2_TML_out_pres_bar,
            in_pres_bar = LH2_TML_in_pres_bar,
            fluid_flow_kg_per_sec = TML_H2_flow_kg_per_sec, 
            dens_kg_per_cu_m = dens_liq_H2_kg_per_cu_m
            )
    
    # calculate loading pump energy (kWh/kg H2)
    LH2_TML_pump_elec_kWh_per_kg = \
        LH2_TML_pump_tot_power_kW / tot_H2_deliv_kg_per_hr
    
    # convert loading pump energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_TML_pump_elec_MJ_per_MJ = \
        LH2_TML_pump_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        LH2_TML_pump_elec_kWh_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        LH2_TML_pump_elec_MJ_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # low-head truck loading pump energy emissions

    # calculate loading pump energy emissions (kg CO2/kg H2)
    LH2_TML_pump_ghg_kg_CO2_per_kg = \
        LH2_TML_pump_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert loading pump energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_TML_pump_ghg_g_CO2_per_MJ = \
        LH2_TML_pump_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        LH2_TML_pump_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        LH2_TML_pump_ghg_g_CO2_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # low-head truck loading pump energy cost
    
    # calculate loading pump energy cost ($/kg H2)
    LH2_TML_pump_elec_cost_usd_per_kg = \
        LH2_TML_pump_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate loading pump energy cost ($/yr)
    LH2_TML_pump_elec_cost_usd_per_yr = \
        LH2_TML_pump_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        LH2_TML_pump_elec_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        LH2_TML_pump_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # low-head truck loading pump installed cost and annual O&M cost
            
    # calculate hydrogen volumetric flowrate through loading pump (m^3/hr)
    LH2_TML_H2_flow_cu_m_per_hr = \
        TML_H2_flow_kg_per_day / dens_liq_H2_kg_per_cu_m / hr_per_day

    # calculate loading pump installed cost ($) and annual 
    # O&M cost ($/yr), both in output dollar year
    LH2_TML_pump_inst_cost_usd, \
    LH2_TML_pump_om_cost_usd_per_yr, \
    LH2_TML_pump_dollar_year = \
        low_head_pump_fixed_costs(
            num_pumps = LH2_TML_num_pumps, 
            fluid_flow_cu_m_per_hr = LH2_TML_H2_flow_cu_m_per_hr,
            output_dollar_year = output_dollar_year
            )
    
    # calculate loading pump O&M cost ($/kg H2)
    LH2_TML_pump_om_cost_usd_per_kg = \
        LH2_TML_pump_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        LH2_TML_pump_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        LH2_TML_pump_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # terminal storage installed cost and annual O&M cost
        
    # calculate total storage capacity (m^3) and number of 
    # storage tanks required at liquid hydrogen terminal
    LH2_TML_stor_tank_capacity_cu_m, \
    LH2_TML_num_tanks = \
        LH2_terminal_storage_size(
            H2_flow_kg_per_day = LH2_TML_H2_in_flow_kg_per_day,
            stor_amt_days = LH2_TML_stor_amt_days
            )
    
    # calculate storage installed cost ($) and annual O&M cost ($/yr), 
    # both in output dollar year
    LH2_TML_stor_inst_cost_usd, \
    LH2_TML_stor_om_cost_usd_per_yr, \
    LH2_TML_stor_dollar_year = \
        LH2_terminal_storage_fixed_costs(
            stor_tank_capacity_cu_m = LH2_TML_stor_tank_capacity_cu_m,
            num_tanks = LH2_TML_num_tanks,
            output_dollar_year = output_dollar_year
            )
                
    # calculate storage O&M cost ($/kg H2)
    LH2_TML_stor_om_cost_usd_per_kg = \
        LH2_TML_stor_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        LH2_TML_stor_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        LH2_TML_stor_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # terminal total capital investment and "other" annual O&M costs
    
    # "Other" O&M costs include insurance, property taxes, licensing and 
    # permits. Operation, maintenance, and repair costs for individual 
    # terminal components (e.g., compressor) are calculated separately.
    
    # calculate terminal total initial capital investment ($)
    # (= loading pump and tank installed cost for now)
    LH2_TML_init_cap_inv_usd = \
        LH2_TML_pump_inst_cost_usd + \
        LH2_TML_stor_inst_cost_usd
        
    # calculate terminal cost allocations (%) to loading pump and tanks
    # % of terminal total initial capital investment
    # use to allocate total capital investment, other O&M costs, and labor 
    # cost
    LH2_TML_pump_cost_perc = \
        LH2_TML_pump_inst_cost_usd / \
        LH2_TML_init_cap_inv_usd
    LH2_TML_stor_cost_perc = \
        LH2_TML_stor_inst_cost_usd / \
        LH2_TML_init_cap_inv_usd
        
    # check whether cost allocations (%) sum to one
    # raise error if false
    if abs(
            LH2_TML_pump_cost_perc + \
            LH2_TML_stor_cost_perc - \
            1.0
            ) >= 1.0e-9:
        raise ValueError(
            'Component cost allocations need to sum to one.'
            )
        
    # check if all terminal components have the same dollar year
    # if true, assign dollar year of refueling station costs to the dollar 
    # year of one of the components 
    if (LH2_TML_pump_dollar_year == \
        LH2_TML_stor_dollar_year):
        LH2_TML_dollar_year = LH2_TML_stor_dollar_year
    else:
        raise ValueError(
            'Dollar year of components need to match.'
            )

    # calculate terminal total capital investment ($, output dollar year) 
    LH2_TML_tot_cap_inv_usd, \
    LH2_TML_cap_cost_dollar_year = \
        non_station_total_capital_investment(
            init_cap_inv_usd = LH2_TML_init_cap_inv_usd, 
            input_dollar_year = LH2_TML_dollar_year
            )
    
    # calculate terminal "other" annual O&M costs ($/yr, output dollar year)
    LH2_TML_om_cost_usd_per_yr, \
    LH2_TML_om_cost_dollar_year = \
        other_om_cost(
            tot_cap_inv_usd = LH2_TML_tot_cap_inv_usd,
            input_dollar_year = LH2_TML_cap_cost_dollar_year
            )
    
    # calculate terminal "other" O&M costs ($/kg H2)
    LH2_TML_om_cost_usd_per_kg = \
        LH2_TML_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign "other" O&M costs to loading pump and tanks
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        LH2_TML_om_cost_usd_per_yr * \
            LH2_TML_pump_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        LH2_TML_om_cost_usd_per_kg * \
            LH2_TML_pump_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        LH2_TML_om_cost_usd_per_yr * \
            LH2_TML_stor_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        LH2_TML_om_cost_usd_per_kg * \
            LH2_TML_stor_cost_perc
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: terminal annual labor cost 

    # calculate terminal annual labor cost ($/yr, output dollar year), 
    # including overhead and G&A
    LH2_TML_labor_cost_usd_per_yr, \
    LH2_TML_labor_cost_dollar_year = \
        non_station_labor_cost(
            H2_flow_kg_per_day = TML_H2_flow_kg_per_day, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate terminal labor cost ($/kg H2)
    LH2_TML_labor_cost_usd_per_kg = \
        LH2_TML_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign labor cost to storagen pump and tanks
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        LH2_TML_labor_cost_usd_per_yr * \
            LH2_TML_pump_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        LH2_TML_labor_cost_usd_per_kg * \
            LH2_TML_pump_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        LH2_TML_labor_cost_usd_per_yr * \
            LH2_TML_stor_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        LH2_TML_labor_cost_usd_per_kg * \
            LH2_TML_stor_cost_perc
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # low-head truck loading pump levelized capital cost
    
    # calculate loading pump total capital investment ($) 
    # (= terminal total capital investment allocated to loading pump)
    LH2_TML_pump_tot_cap_inv_usd = \
        LH2_TML_pump_cost_perc * LH2_TML_tot_cap_inv_usd
    
    # calculate loading pump levelized capital cost 
    # ($/yr, output dollar year)
    LH2_TML_pump_lev_cap_cost_usd_per_yr, \
    LH2_TML_pump_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = LH2_TML_pump_tot_cap_inv_usd, 
            life_yr = TML_pump_life_yr, 
            depr_yr = TML_pump_depr_yr,
            input_dollar_year = LH2_TML_cap_cost_dollar_year
            )
    
    # calculate loading pump levelized capital cost ($/kg H2)
    LH2_TML_pump_lev_cap_cost_usd_per_kg = \
        LH2_TML_pump_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        LH2_TML_pump_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        LH2_TML_pump_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - liquid hydrogen: 
    # storage levelized capital cost 

    # calculate storage total capital investment ($) 
    # (= terminal total capital investment allocated to storage)
    LH2_TML_stor_tot_cap_inv_usd = \
        LH2_TML_stor_cost_perc * LH2_TML_tot_cap_inv_usd
    
    # calculate storage levelized capital cost 
    # ($/yr, output dollar year)
    LH2_TML_stor_lev_cap_cost_usd_per_yr, \
    LH2_TML_stor_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = LH2_TML_stor_tot_cap_inv_usd, 
            life_yr = TML_stor_life_yr, 
            depr_yr = TML_stor_depr_yr,
            input_dollar_year = LH2_TML_cap_cost_dollar_year
            )
    
    # calculate storage levelized capital cost ($/kg H2)
    LH2_TML_stor_lev_cap_cost_usd_per_kg = \
        LH2_TML_stor_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'cryogenic storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        LH2_TML_stor_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'cryogenic storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        LH2_TML_stor_lev_cap_cost_usd_per_kg
        ])
        
    #%% CALCULATIONS: LIQUID HYDROGEN DELIVERY ("LH2")
    # TRANSPORT @ TRUCK

    # ------------------------------------------------------------------------
    # transport - liquid hydrogen: 
    # truck fuel consumption, number of trucks required, number of deliveries 
    # per day, total trip time
    
    # calculate truck fuel consumption (gallon/kg H2), number of trucks, 
    # number of deliveries per day, total trip time (hr/trip)
    # TODO: revisit gal/kg H2 when adding losses (transported vs. delivered)
    LH2_truck_fuel_gal_per_kg, \
    LH2_num_trucks, \
    LH2_truck_num_delivs_per_day, \
    LH2_truck_trip_time_hr = \
        transport_energy(
            deliv_dist_mi = deliv_dist_mi_rt, 
            speed_mi_per_hr = truck_speed_mi_per_hr, 
            load_unload_time_hr = liq_truck_load_unload_time_hr,
            fuel_econ_mi_per_gal = truck_fuel_econ_mi_per_gal,
            deliv_capacity_kg = liq_truck_deliv_capacity_kgH2, 
            cargo_flow_kg_per_day = TML_H2_flow_kg_per_day
            )
    
    # convert truck fuel consumption to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_truck_fuel_MJ_per_MJ = \
        LH2_truck_fuel_gal_per_kg * low_heat_val_diesel_MJ_per_gal / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy consumption', 
        'diesel consumption', 
        'gallon/kg H2', 
        LH2_truck_fuel_gal_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy consumption', 
        'diesel consumption', 
        'MJ/MJ H2 (LHV)', 
        LH2_truck_fuel_MJ_per_MJ
        ])
            
    # ------------------------------------------------------------------------
    # transport - liquid hydrogen: 
    # truck fuel emissions

    # calculate truck fuel emissions (kg CO2/kg H2)
    LH2_truck_ghg_kg_CO2_per_kg = \
        LH2_truck_fuel_gal_per_kg * diesel_ghg_kg_CO2_per_gal
    
    # convert truck fuel emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_truck_ghg_g_CO2_per_MJ = \
        LH2_truck_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        LH2_truck_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        LH2_truck_ghg_g_CO2_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # transport - liquid hydrogen: truck fuel cost
    
    # calculate truck fuel cost ($/kg H2)
    LH2_truck_fuel_cost_usd_per_kg = \
        LH2_truck_fuel_gal_per_kg * diesel_cost_usd_per_gal
    
    # calculate truck fuel cost ($/yr)
    LH2_truck_fuel_cost_usd_per_yr = \
        LH2_truck_fuel_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy cost', 
        'fuel cost', 
        '$/yr', 
        LH2_truck_fuel_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy cost', 
        'fuel cost', 
        '$/kg H2', 
        LH2_truck_fuel_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # transport - liquid hydrogen: truck capital cost and annual O&M cost
    
    # calculate liquid hydrogen truck capital cost ($, output dollar year) 
    LH2_truck_cap_cost_usd, \
    LH2_truck_cap_cost_dollar_year = \
        liquid_truck_capital_cost(
            num_trucks = LH2_num_trucks,
            output_dollar_year = output_dollar_year
            )
    
    # calculate number of deliveries (truck-trips) per year for liquid 
    # hydrogen
    LH2_truck_num_delivs_per_yr = \
        LH2_truck_num_delivs_per_day * day_per_yr 
    
    # calculate truck annual O&M cost ($/yr, output dollar year)
    LH2_truck_om_cost_usd_per_yr, \
    LH2_truck_om_cost_dollar_year = \
        truck_om_cost(
            deliv_dist_mi = deliv_dist_mi_rt, 
            num_delivs_per_yr = LH2_truck_num_delivs_per_yr, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate truck O&M cost ($/kg H2)
    LH2_truck_om_cost_usd_per_kg = \
        LH2_truck_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        LH2_truck_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        LH2_truck_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # transport - liquid hydrogen: truck annual labor cost
    
    # calculate truck annual labor cost ($/yr, output dollar year), 
    # including overhead and G&A
    LH2_truck_labor_cost_usd_per_yr, \
    LH2_truck_labor_cost_dollar_year = \
        truck_labor_cost(
            num_delivs_per_yr = LH2_truck_num_delivs_per_yr, 
            trip_time_hr = LH2_truck_trip_time_hr, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate truck labor cost ($/kg H2)
    LH2_truck_labor_cost_usd_per_kg = \
        LH2_truck_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        LH2_truck_labor_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        LH2_truck_labor_cost_usd_per_kg
        ])
        
    # ------------------------------------------------------------------------
    # transport - liquid hydrogen: truck levelized capital cost
        
    # calculate truck levelized capital cost ($/yr, output dollar year)
    # truck total capital investment = truck capital cost
    LH2_truck_lev_cap_cost_usd_per_yr, \
    LH2_truck_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = LH2_truck_cap_cost_usd, 
            life_yr = truck_life_yr, 
            depr_yr = truck_depr_yr,
            input_dollar_year = LH2_truck_cap_cost_dollar_year
            )
    
    # calculate truck levelized capital cost ($/kg H2)
    LH2_truck_lev_cap_cost_usd_per_kg = \
        LH2_truck_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        LH2_truck_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        LH2_truck_lev_cap_cost_usd_per_kg
        ])
        
    #%% CALCULATIONS: LIQUID HYDROGEN DELIVERY ("LH2")
    # RECONDITIONING @ REFUELING STATION

    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station cryogenic storage installed cost and annual O&M cost
    
    # calculate total storage capacity required (kg) at liquid hydrogen 
    # refueling station
    LH2_STN_cryo_stor_tot_capacity_kg = \
        LH2_station_cryo_storage_size(
            stn_capacity_kg_per_day = target_stn_capacity_kg_per_day,
            truck_load_kg = liq_truck_deliv_capacity_kgH2,
            ) 
        
    # calculate storage installed cost ($) and annual O&M cost ($/yr)
    # per station, both in output dollar year
    LH2_STN_cryo_stor_inst_cost_usd_per_stn, \
    LH2_STN_cryo_stor_om_cost_usd_per_yr_per_stn, \
    LH2_STN_cryo_stor_dollar_year = \
        LH2_station_cryo_storage_fixed_costs(
            stor_tot_capacity_kg = LH2_STN_cryo_stor_tot_capacity_kg, 
            output_dollar_year = output_dollar_year
            )
                        
    # calculate storage installed cost ($) 
    # sum of all stations
    LH2_STN_cryo_stor_inst_cost_usd = \
        LH2_STN_cryo_stor_inst_cost_usd_per_stn * target_num_stns

    # calculate storage O&M cost ($/yr)
    # sum of all stations
    LH2_STN_cryo_stor_om_cost_usd_per_yr = \
        LH2_STN_cryo_stor_om_cost_usd_per_yr_per_stn * target_num_stns

    # calculate storage O&M cost ($/kg H2)
    LH2_STN_cryo_stor_om_cost_usd_per_kg = \
        LH2_STN_cryo_stor_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        LH2_STN_cryo_stor_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        LH2_STN_cryo_stor_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station cryogenic pump energy consumption and size
        
    # calculate cryogenic pump inlet pressure (bar)
    # HDSAM V3.1: hydrogen supply pressure from dewar
    LH2_STN_in_pres_bar = \
        LH2_STN_in_pres_atm * Pa_per_atm / Pa_per_bar
    
    # calculate cryogenic pump power (kW) and size (kW/pump) 
    # at refueling station
    LH2_STN_pump_tot_power_kW, \
    LH2_STN_pump_power_kW_per_pump, \
    LH2_STN_num_pumps = \
        pump_power_and_size(
            out_pres_bar = LH2_STN_out_pres_bar,
            in_pres_bar = LH2_STN_in_pres_bar,
            fluid_flow_kg_per_sec = STN_H2_flow_kg_per_sec, 
            dens_kg_per_cu_m = dens_liq_H2_kg_per_cu_m
            )
    
    # calculate cryogenic pump energy (kWh/kg H2)
    LH2_STN_pump_elec_kWh_per_kg = \
        LH2_STN_pump_tot_power_kW * target_num_stns / \
        tot_H2_deliv_kg_per_hr
    
    # convert cryogenic pump energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_STN_pump_elec_MJ_per_MJ = \
        LH2_STN_pump_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        LH2_STN_pump_elec_kWh_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        LH2_STN_pump_elec_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station cryogenic pump energy emissions

    # calculate cryogenic pump energy emissions (kg CO2/kg H2)
    LH2_STN_pump_ghg_kg_CO2_per_kg = \
        LH2_STN_pump_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert cryogenic pump energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    LH2_STN_pump_ghg_g_CO2_per_MJ = \
        LH2_STN_pump_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        LH2_STN_pump_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        LH2_STN_pump_ghg_g_CO2_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station cryogenic pump energy cost
    
    # calculate refueling station cryogenic pump energy cost ($/kg H2)
    LH2_STN_pump_elec_cost_usd_per_kg = \
        LH2_STN_pump_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate refueling station cryogenic pump energy cost ($/yr)
    LH2_STN_pump_elec_cost_usd_per_yr = \
        LH2_STN_pump_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        LH2_STN_pump_elec_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        LH2_STN_pump_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station cryogenic pump installed cost and annual O&M cost
    
    # calculate refueling station cryogenic pump installed cost ($) and annual 
    # O&M cost ($/yr) per station, both in output dollar year
    LH2_STN_pump_inst_cost_usd_per_stn, \
    LH2_STN_pump_om_cost_usd_per_yr_per_stn, \
    LH2_STN_pump_dollar_year = \
        cryo_pump_fixed_costs(
            num_pumps = LH2_STN_num_pumps, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate refueling station cryogenic pump installed cost ($) 
    # sum of all stations
    LH2_STN_pump_inst_cost_usd = \
        LH2_STN_pump_inst_cost_usd_per_stn * target_num_stns

    # calculate refueling station cryogenic pump O&M cost ($/yr)
    # sum of all stations
    LH2_STN_pump_om_cost_usd_per_yr = \
        LH2_STN_pump_om_cost_usd_per_yr_per_stn * target_num_stns

    # calculate refueling station cryogenic pump O&M cost ($/kg H2)
    LH2_STN_pump_om_cost_usd_per_kg = \
        LH2_STN_pump_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        LH2_STN_pump_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        LH2_STN_pump_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station vaporizer installed cost and annual O&M cost

    # calculate hydrogen mass flowrate (kg/hr) at refueling station
    LH2_STN_H2_flow_kg_per_hr = \
        target_stn_capacity_kg_per_day / hr_per_day
    
    # calculate refueling station vaporizer installed cost ($) and annual O&M 
    # cost ($/yr) per station, both in output dollar year 
    LH2_STN_vap_inst_cost_usd_per_stn, \
    LH2_STN_vap_om_cost_usd_per_yr_per_stn, \
    LH2_STN_vap_dollar_year = \
        vaporizer_fixed_costs(
            fluid_flow_kg_per_hr = LH2_STN_H2_flow_kg_per_hr,
            output_dollar_year = output_dollar_year
            )
    
    # calculate refueling station vaporizer installed cost ($) 
    # sum of all stations
    LH2_STN_vap_inst_cost_usd = \
        LH2_STN_vap_inst_cost_usd_per_stn * target_num_stns

    # calculate refueling station vaporizer O&M cost ($/yr)
    # sum of all stations
    LH2_STN_vap_om_cost_usd_per_yr = \
        LH2_STN_vap_om_cost_usd_per_yr_per_stn * target_num_stns

    # calculate refueling station vaporizer O&M cost ($/kg H2)
    LH2_STN_vap_om_cost_usd_per_kg = \
        LH2_STN_vap_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'vaporization', 
        'vaporizer', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        LH2_STN_vap_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'vaporization', 
        'vaporizer', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        LH2_STN_vap_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station cascade storage installed cost and annual O&M cost

    # calculate total cascade storage capacity required (kg) at
    # refueling station
    LH2_STN_casc_stor_tot_capacity_kg = \
        station_cascade_storage_size(
            stn_capacity_kg_per_day = target_stn_capacity_kg_per_day,
            casc_stor_size_frac = LH2_STN_casc_stor_size_frac
            ) 
        
    # calculate cascade storage installed cost ($) and annual O&M 
    # cost ($/yr) per station, both in output dollar year
    LH2_STN_casc_stor_inst_cost_usd_per_stn, \
    LH2_STN_casc_stor_om_cost_usd_per_yr_per_stn, \
    LH2_STN_casc_stor_dollar_year = \
        station_cascade_storage_fixed_costs(
            stor_tot_capacity_kg = LH2_STN_casc_stor_tot_capacity_kg, 
            output_dollar_year = output_dollar_year
            )
                        
    # calculate cascade storage installed cost ($)
    # sum of all stations
    LH2_STN_casc_stor_inst_cost_usd = \
        LH2_STN_casc_stor_inst_cost_usd_per_stn * target_num_stns

    # calculate cascade storage O&M cost ($/yr)
    # sum of all stations
    LH2_STN_casc_stor_om_cost_usd_per_yr = \
        LH2_STN_casc_stor_om_cost_usd_per_yr_per_stn * target_num_stns

    # calculate cascade storage O&M cost ($/kg H2)
    LH2_STN_casc_stor_om_cost_usd_per_kg = \
        LH2_STN_casc_stor_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        LH2_STN_casc_stor_om_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        LH2_STN_casc_stor_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station total capital investment and "other" annual O&M costs
    
    # "Other" O&M costs include insurance, property taxes, licensing and 
    # permits. Operation, maintenance, and repair costs for individual 
    # refueling station components (e.g., pump) are calculated separately.
    
    # calculate refueling station total initial capital investment ($)
    # (= cryogenic storage + cryogenic pump + vaporizer + cascade storage 
    # for *liquid* hydrogen refueling station)
    # sum of all stations
    LH2_STN_init_cap_inv_usd = \
        LH2_STN_cryo_stor_inst_cost_usd + \
        LH2_STN_pump_inst_cost_usd + \
        LH2_STN_vap_inst_cost_usd + \
        LH2_STN_casc_stor_inst_cost_usd
    
    # calculate refueling station cost allocations (%) to cryogenic storage, 
    # cryogenic pump, vaporizer, and cascade storage
    # % of refueling station total initial capital investment
    # use to allocate total capital investment, other O&M costs, and labor 
    # cost
    LH2_STN_cryo_stor_cost_perc = \
        LH2_STN_cryo_stor_inst_cost_usd / \
        LH2_STN_init_cap_inv_usd
    LH2_STN_pump_cost_perc = \
        LH2_STN_pump_inst_cost_usd / \
        LH2_STN_init_cap_inv_usd
    LH2_STN_vap_cost_perc = \
        LH2_STN_vap_inst_cost_usd / \
        LH2_STN_init_cap_inv_usd
    LH2_STN_casc_stor_cost_perc = \
        LH2_STN_casc_stor_inst_cost_usd / \
        LH2_STN_init_cap_inv_usd
    
    # check whether cost allocations (%) sum to one
    # raise error if false
    if abs(
            LH2_STN_cryo_stor_cost_perc + \
            LH2_STN_pump_cost_perc + \
            LH2_STN_vap_cost_perc + \
            LH2_STN_casc_stor_cost_perc - \
            1.0
            ) >= 1.0e-9:
        raise ValueError(
            'Component cost allocations need to sum to one.'
            )
    
    # check if all refueling station components have the same dollar year
    # if true, assign dollar year of refueling station costs to the dollar 
    # year of one of the components
    if (LH2_STN_cryo_stor_dollar_year == \
        LH2_STN_pump_dollar_year) \
        and (LH2_STN_pump_dollar_year == \
             LH2_STN_vap_dollar_year) \
        and (LH2_STN_vap_dollar_year == \
             LH2_STN_casc_stor_dollar_year):
        LH2_STN_dollar_year = LH2_STN_cryo_stor_dollar_year
    else:
        raise ValueError(
            'Dollar year of components need to match.'
            )
    
    # calculate refueling station total capital investment 
    # ($, output dollar year), sum of all stations
    LH2_STN_tot_cap_inv_usd, \
    LH2_STN_cap_cost_dollar_year = \
        station_total_capital_investment(
            init_cap_inv_usd = LH2_STN_init_cap_inv_usd, 
            input_dollar_year = LH2_STN_dollar_year
            )
    
    # calculate refueling station "other" annual O&M costs 
    # ($/yr, output dollar year), sum of all stations
    LH2_STN_om_cost_usd_per_yr, \
    LH2_STN_om_cost_dollar_year = \
        other_om_cost(
            tot_cap_inv_usd = LH2_STN_tot_cap_inv_usd,
            input_dollar_year = LH2_STN_cap_cost_dollar_year
            )
    
    # calculate refueling station "other" O&M costs ($/kg H2)
    LH2_STN_om_cost_usd_per_kg = \
        LH2_STN_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign "other" O&M costs to cryogenic storage, cryogenic pump, 
    # vaporizer, and cascade storage
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        LH2_STN_om_cost_usd_per_yr * \
            LH2_STN_cryo_stor_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cryogenic storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        LH2_STN_om_cost_usd_per_kg * \
            LH2_STN_cryo_stor_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping',  
        'cryogenic pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        LH2_STN_om_cost_usd_per_yr * \
            LH2_STN_pump_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping',  
        'cryogenic pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        LH2_STN_om_cost_usd_per_kg * \
            LH2_STN_pump_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'vaporization',  
        'vaporizer', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        LH2_STN_om_cost_usd_per_yr * \
            LH2_STN_vap_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'vaporization',  
        'vaporizer', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        LH2_STN_om_cost_usd_per_kg * \
            LH2_STN_vap_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage',  
        'cascade storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        LH2_STN_om_cost_usd_per_yr * \
            LH2_STN_casc_stor_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage',  
        'cascade storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        LH2_STN_om_cost_usd_per_kg * \
            LH2_STN_casc_stor_cost_perc
        ])

    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: refueling station annual labor cost 
    
    # calculate refueling station annual labor cost 
    # ($/yr, output dollar year) per station, including overhead and G&A
    LH2_STN_labor_cost_usd_per_yr_per_stn, \
    LH2_STN_labor_cost_dollar_year = \
        station_labor_cost(
            H2_flow_kg_per_day = target_stn_capacity_kg_per_day, 
            output_dollar_year = output_dollar_year
            )
        
    # calculate refueling station labor cost ($/yr)
    # sum of all stations
    LH2_STN_labor_cost_usd_per_yr = \
        LH2_STN_labor_cost_usd_per_yr_per_stn * target_num_stns
    
    # calculate refueling station labor cost ($/kg H2)
    LH2_STN_labor_cost_usd_per_kg = \
        LH2_STN_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign labor to cryogenic torage, cryogenic pump, vaporizer, and
    # cascade storage
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage',  
        'cryogenic storage', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        LH2_STN_labor_cost_usd_per_yr * \
            LH2_STN_cryo_stor_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage',  
        'cryogenic storage', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        LH2_STN_labor_cost_usd_per_kg * \
            LH2_STN_cryo_stor_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        LH2_STN_labor_cost_usd_per_yr * \
            LH2_STN_pump_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        LH2_STN_labor_cost_usd_per_kg * \
            LH2_STN_pump_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'vaporization', 
        'vaporizer', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        LH2_STN_labor_cost_usd_per_yr * \
            LH2_STN_vap_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'vaporization', 
        'vaporizer', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        LH2_STN_labor_cost_usd_per_kg * \
            LH2_STN_vap_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        LH2_STN_labor_cost_usd_per_yr * \
            LH2_STN_casc_stor_cost_perc
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        LH2_STN_labor_cost_usd_per_kg * \
            LH2_STN_casc_stor_cost_perc
        ])

    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station *cryogenic* storage levelized capital cost
        
    # calculate refueling station cryogenic storage total capital 
    # investment ($) 
    # (= refueling station total capital investment allocated to 
    # cryogenic storage)
    # sum of all stations
    LH2_STN_cryo_stor_tot_cap_inv_usd = \
        LH2_STN_cryo_stor_cost_perc * LH2_STN_tot_cap_inv_usd
    
    # calculate refueling station cryogenic storage levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    LH2_STN_cryo_stor_lev_cap_cost_usd_per_yr, \
    LH2_STN_cryo_stor_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = LH2_STN_cryo_stor_tot_cap_inv_usd, 
            life_yr = STN_stor_life_yr, 
            depr_yr = STN_stor_depr_yr,
            input_dollar_year = LH2_STN_dollar_year
            )
    
    # calculate refueling station cryogenic storage levelized capital cost 
    # ($/kg H2)
    LH2_STN_cryo_stor_lev_cap_cost_usd_per_kg = \
        LH2_STN_cryo_stor_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cryogenic storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        LH2_STN_cryo_stor_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cryogenic storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        LH2_STN_cryo_stor_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station cryogenic pump levelized capital cost
        
    # calculate refueling station cryogenic pump total capital investment ($) 
    # (= refueling station total capital investment allocated to cryogenic
    # pump)
    # sum of all stations
    LH2_STN_pump_tot_cap_inv_usd = \
        LH2_STN_pump_cost_perc * LH2_STN_tot_cap_inv_usd
    
    # calculate refueling station cryogenic pump levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    LH2_STN_pump_lev_cap_cost_usd_per_yr, \
    LH2_STN_pump_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = LH2_STN_pump_tot_cap_inv_usd, 
            life_yr = STN_pump_life_yr, 
            depr_yr = STN_pump_depr_yr,
            input_dollar_year = LH2_STN_dollar_year
            )
    
    # calculate refueling station cryogenic pump levelized capital cost 
    # ($/kg H2)
    LH2_STN_pump_lev_cap_cost_usd_per_kg = \
        LH2_STN_pump_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        LH2_STN_pump_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'cryogenic pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        LH2_STN_pump_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station vaporizer levelized capital cost
        
    # calculate refueling station vaporizer total capital investment ($) 
    # (= refueling station total capital investment allocated to vaporizer)
    # sum of all stations
    LH2_STN_vap_tot_cap_inv_usd = \
        LH2_STN_vap_cost_perc * LH2_STN_tot_cap_inv_usd
    
    # calculate refueling station vaporizer levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    LH2_STN_vap_lev_cap_cost_usd_per_yr, \
    LH2_STN_vap_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = LH2_STN_vap_tot_cap_inv_usd, 
            life_yr = STN_vap_life_yr, 
            depr_yr = STN_vap_depr_yr,
            input_dollar_year = LH2_STN_dollar_year
            )
    
    # calculate refueling station vaporizer levelized capital cost ($/kg H2)
    LH2_STN_vap_lev_cap_cost_usd_per_kg = \
        LH2_STN_vap_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'vaporization', 
        'vaporizer', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        LH2_STN_vap_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'vaporization', 
        'vaporizer', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        LH2_STN_vap_lev_cap_cost_usd_per_kg
        ])
        
    # ------------------------------------------------------------------------
    # reconditioning - liquid hydrogen: 
    # refueling station *cascade* storage levelized capital cost
        
    # calculate refueling station cascade storage total capital investment ($) 
    # (= refueling station total capital investment allocated to cascade 
    # storage)
    # sum of all stations
    LH2_STN_casc_stor_tot_cap_inv_usd = \
        LH2_STN_casc_stor_cost_perc * LH2_STN_tot_cap_inv_usd
    
    # calculate refueling station cascade storage levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    LH2_STN_casc_stor_lev_cap_cost_usd_per_yr, \
    LH2_STN_casc_stor_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = LH2_STN_casc_stor_tot_cap_inv_usd, 
            life_yr = STN_stor_life_yr, 
            depr_yr = STN_stor_depr_yr,
            input_dollar_year = LH2_STN_dollar_year
            )
    
    # calculate refueling station cascade storage levelized capital cost 
    # ($/kg H2)
    LH2_STN_casc_stor_lev_cap_cost_usd_per_kg = \
        LH2_STN_casc_stor_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        LH2_STN_casc_stor_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'liquid hydrogen', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        LH2_STN_casc_stor_lev_cap_cost_usd_per_kg
        ])
    
    #%% CALCULATIONS: LOHC / FORMIC ACID DELIVERY ("FAH2")
    # FORMIC ACID PRODUCTION @ TERMINAL
    
    # ------------------------------------------------------------------------
    # production - formic acid: formic acid purchase cost
    
    # initialize formic acid purchase cost (zero by default)
    FAH2_TML_FA_purc_cost_usd_per_yr = 0.0
    
    # if purchase formic acid, calculate purchase costs
    if FA_prod_pathway == 'purchase':
    
        # calculate formic acid purchase cost ($/yr)   
        FAH2_TML_FA_purc_cost_usd_per_yr = \
            purc_cost_FA_usd_per_kg * FAH2_TML_FA_flow_kg_per_day * \
            day_per_yr
    
    # calculate formic acid production cost ($/kg H2)   
    FAH2_TML_FA_purc_cost_usd_per_kg = \
        FAH2_TML_FA_purc_cost_usd_per_yr / tot_H2_deliv_kg_per_yr

    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'purchase', 
        'formic acid purchase', 
        'O&M cost', 
        'purchase cost', 
        '$/yr', 
        FAH2_TML_FA_purc_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'purchase', 
        'formic acid purchase', 
        'O&M cost', 
        'purchase cost', 
        '$/kg H2', 
        FAH2_TML_FA_purc_cost_usd_per_kg
        ])
   
    # ------------------------------------------------------------------------
    # production - formic acid: emissions of purchased formic acid

    # initialize emissions of purchased formic acid (zero by default)
    FAH2_TML_FA_purc_ghg_kg_CO2_per_kg = 0.0

    # if purchase formic acid, calculate emissions of purchased
    # formic acid
    if FA_prod_pathway == 'purchase':
        
        # calculate emissions of purchased formic acid (kg CO2-eq/kg H2)
        FAH2_TML_FA_purc_ghg_kg_CO2_per_kg = \
            FA_prod_ghg_kg_CO2_per_kg * FAH2_TML_FA_flow_kg_per_day * \
            day_per_yr / tot_H2_deliv_kg_per_yr    
    
    # convert emissions of purchased formic acid to g CO2-eq/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_TML_FA_purc_ghg_g_CO2_per_MJ = \
        FAH2_TML_FA_purc_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'purchase', 
        'formic acid purchase', 
        'emissions', 
        'upstream emissions', 
        'kg CO2/kg H2', 
        FAH2_TML_FA_purc_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'purchase', 
        'formic acid purchase', 
        'emissions', 
        'upstream emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_TML_FA_purc_ghg_g_CO2_per_MJ
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: hydrogen purchase cost
    
    # initialize hydrogen purchase cost (zero by default)
    FAH2_TML_H2_purc_cost_usd_per_yr = 0.0
    
    # if produce formic acid at terminal (as opposed to purchase), 
    # calculate hydrogen purchase costs
    if FA_prod_pathway != 'purchase':
    
        # calculate hydrogen purchase cost ($/yr)   
        FAH2_TML_H2_purc_cost_usd_per_yr = \
            purc_cost_H2_usd_per_kg * FAH2_TML_H2_flow_kg_per_day * \
            day_per_yr
    
    # calculate hydrogen production cost ($/kg H2)   
    FAH2_TML_H2_purc_cost_usd_per_kg = \
        FAH2_TML_H2_purc_cost_usd_per_yr / tot_H2_deliv_kg_per_yr

    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'O&M cost', 
        'purchase cost', 
        '$/yr', 
        FAH2_TML_H2_purc_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'O&M cost', 
        'purchase cost', 
        '$/kg H2', 
        FAH2_TML_H2_purc_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: emissions of purchased hydrogen
    
    # initialize emissions of purchased hydrogen (zero by default)
    FAH2_TML_H2_purc_ghg_kg_CO2_per_kg = 0.0

    # if produce formic acid at terminal (as opposed to purchase), 
    # calculate emissions of purchased hydrogen
    if FA_prod_pathway != 'purchase':
        
        # calculate emissions of purchased hydrogen (kg CO2-eq/kg H2)
        FAH2_TML_H2_purc_ghg_kg_CO2_per_kg = \
            H2_prod_ghg_kg_CO2_per_kg * FAH2_TML_H2_flow_kg_per_day * \
            day_per_yr / tot_H2_deliv_kg_per_yr    
                
    # convert emissions of purchased hydrogen to g CO2-eq/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_TML_H2_purc_ghg_g_CO2_per_MJ = \
        FAH2_TML_H2_purc_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'emissions', 
        'upstream emissions', 
        'kg CO2/kg H2', 
        FAH2_TML_H2_purc_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'purchase', 
        'hydrogen purchase', 
        'emissions', 
        'upstream emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_TML_H2_purc_ghg_g_CO2_per_MJ
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation hydrogen compressor energy consumption and size
        
    # initialize hydrogenation hydrogen compressor power (kW) 
    # and size (kW/stage) (zero by default)
    FAH2_TML_hydr_compr_tot_power_kW = 0.0
    FAH2_TML_hydr_compr_power_kW_per_stg = 0.0
    FAH2_TML_hydr_compr_num_stgs = 0
    
    if FA_prod_pathway == 'thermo':

        # calculate hydrogenation hydrogen compressor power (kW) 
        # and size (kW/stage) at terminal
        # compressor outlet pressure = hydrogenation reaction pressure
        # TODO: revisit compressibility 
        # (for now, assume same as storage compressor at compressed 
        # hydrogen terminal)
        FAH2_TML_hydr_compr_tot_power_kW, \
        FAH2_TML_hydr_compr_power_kW_per_stg, \
        FAH2_TML_hydr_compr_num_stgs = \
            compressor_power_and_size(
                out_pres_bar = FAH2_hydr_pres_bar,
                in_pres_bar = FAH2_TML_in_pres_bar,
                in_temp_K = FAH2_TML_in_temp_K, 
                gas_flow_mol_per_sec = FAH2_TML_H2_flow_mol_per_sec, 
                compressibility = 1.13
                )
                            
    # calculate hydrogenation hydrogen compressor energy (kWh/kg H2)
    FAH2_TML_hydr_compr_elec_kWh_per_kg = \
        FAH2_TML_hydr_compr_tot_power_kW / tot_H2_deliv_kg_per_hr
    
    # convert hydrogenation hydrogen compressor energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_TML_hydr_compr_elec_MJ_per_MJ = \
        FAH2_TML_hydr_compr_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        FAH2_TML_hydr_compr_elec_kWh_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_TML_hydr_compr_elec_MJ_per_MJ
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation hydrogen compressor energy emissions
    
    # calculate hydrogenation hydrogen compressor energy emissions 
    # (kg CO2/kg H2)
    FAH2_TML_hydr_compr_ghg_kg_CO2_per_kg = \
        FAH2_TML_hydr_compr_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert hydrogenation hydrogen compressor energy emissions to 
    # g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_TML_hydr_compr_ghg_g_CO2_per_MJ = \
        FAH2_TML_hydr_compr_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_TML_hydr_compr_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_TML_hydr_compr_ghg_g_CO2_per_MJ
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation hydrogen compressor energy cost
    
    # calculate hydrogenation hydrogen compressor energy cost ($/kg H2)
    FAH2_TML_hydr_compr_elec_cost_usd_per_kg = \
        FAH2_TML_hydr_compr_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate hydrogenation hydrogen compressor energy cost ($/yr)
    FAH2_TML_hydr_compr_elec_cost_usd_per_yr = \
        FAH2_TML_hydr_compr_elec_cost_usd_per_kg * \
        tot_H2_deliv_kg_per_yr
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        FAH2_TML_hydr_compr_elec_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        FAH2_TML_hydr_compr_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------        
    # production - formic acid: 
    # hydrogenation hydrogen compressor installed cost and annual O&M cost
    
    # initialize hydrogenation hydrogen compressor installed cost and 
    # annual O&M cost (zero by default)
    FAH2_TML_hydr_compr_inst_cost_usd = 0.0
    FAH2_TML_hydr_compr_om_cost_usd_per_yr = 0.0
    FAH2_TML_hydr_compr_dollar_year = output_dollar_year
    
    if FA_prod_pathway == 'thermo':

        # calculate hydrogenation hydrogen compressor installed cost ($) and annual O&M 
        # cost ($/yr), both in output dollar year
        FAH2_TML_hydr_compr_inst_cost_usd, \
        FAH2_TML_hydr_compr_om_cost_usd_per_yr, \
        FAH2_TML_hydr_compr_dollar_year = \
            compressor_fixed_costs(
                compr_power_kW_per_stg = \
                    FAH2_TML_hydr_compr_power_kW_per_stg, 
                num_stgs = FAH2_TML_hydr_compr_num_stgs,
                output_dollar_year = output_dollar_year
                )
        
    # calculate hydrogenation hydrogen compressor O&M cost ($/kg H2)
    FAH2_TML_hydr_compr_om_cost_usd_per_kg = \
        FAH2_TML_hydr_compr_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_TML_hydr_compr_om_cost_usd_per_yr
       ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_TML_hydr_compr_om_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation CO2 pump energy consumption and size
        
    # TODO: add hydrogenation CO2 pump energy consumption and size
        
    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation CO2 pump energy emissions

    # TODO: add hydrogenation CO2 pump energy emissions

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation CO2 pump energy cost

    # TODO: add hydrogenation CO2 pump energy cost

    # ------------------------------------------------------------------------        
    # production - formic acid: 
    # hydrogenation CO2 pump installed cost and annual O&M cost

    # TODO: add hydrogenation CO2 installed cost and annual O&M cost
    
    FAH2_TML_hydr_pump_inst_cost_usd = 0.0
    FAH2_TML_hydr_pump_dollar_year = output_dollar_year

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation CO2 vaporizer energy consumption and size
        
    # TODO: add hydrogenation CO2 vaporizer energy consumption and size
        
    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation CO2 vaporizer energy emissions

    # TODO: add hydrogenation CO2 vaporizer energy emissions

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation CO2 vaporizer energy cost

    # TODO: add hydrogenation CO2 vaporizer energy cost

    # ------------------------------------------------------------------------        
    # production - formic acid: 
    # hydrogenation CO2 vaporizer installed cost and annual O&M cost

    # TODO: add hydrogenation CO2 installed cost and annual O&M cost
    
    FAH2_TML_hydr_vap_inst_cost_usd = 0.0
    FAH2_TML_hydr_vap_dollar_year = output_dollar_year

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation reactor energy consumption

    # TODO: add reactor energy consumption
    
    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation reactor energy emissions

    # TODO: add reactor energy emissions
    
    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation reactor energy cost
    
    # TODO: add reactor energy cost

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation reactor installed cost and annual O&M cost
    
    # initialize hydrogenation reactor installed cost and 
    # annual O&M cost (zero by default)
    FAH2_TML_hydr_react_inst_cost_usd = 0.0
    FAH2_TML_hydr_react_om_cost_usd_per_yr = 0.0
    FAH2_TML_hydr_react_dollar_year = output_dollar_year
    
    if FA_prod_pathway == 'thermo':

        # calculate hydrogenation reactor installed cost ($) and annual 
        # O&M cost ($/yr), both in output dollar year
        FAH2_TML_hydr_react_inst_cost_usd, \
        FAH2_TML_hydr_react_om_cost_usd_per_yr, \
        FAH2_TML_hydr_react_dollar_year = \
            reactor_fixed_costs(
                react_pres_bar = FAH2_hydr_pres_bar,
                react_vol_cu_m = FAH2_hydr_react_vol_cu_m, 
                num_reacts = FAH2_num_hydr_reacts, 
                output_dollar_year = output_dollar_year
                )
    
    # calculate hydrogenation reactor O&M cost ($/kg H2)
    FAH2_TML_hydr_react_om_cost_usd_per_kg = \
        FAH2_TML_hydr_react_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_TML_hydr_react_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_TML_hydr_react_om_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # hydrogenation catalyst upfront cost
    
    # initialize hydrogenation catalyst purchase cost (zero by default)
    FAH2_TML_hydr_catal_purc_cost_usd = 0.0
    
    if FA_prod_pathway == 'thermo':

        # calculate hydrogenation catalyst upfront purchase cost ($)
        FAH2_TML_hydr_catal_purc_cost_usd = \
            FAH2_hydr_catal_cost_usd_per_kg * FAH2_hydr_catal_amt_kg

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # CO2 electrolyzer energy consumption and size
    
    # TODO: refine electrolyzer energy consumption and size
    
    # initialize CO2 electrolyzer power (zero by default)
    FAH2_TML_hydr_electr_power_kW = 0.0

    if FA_prod_pathway == 'electro':
        
        # TODO: turn electrolyzer energy calculations into function
        # ideal assumptions for now

        # CO2 electrolyzer energy (kJ/mol formic acid)
        hydr_electr_elec_kJ_per_mol_FA = \
            electr_volt_V * e_per_mol_FA * \
            faraday_const_C_per_mol_e / J_per_kJ
        
        # calculate CO2 electrolyzer power (kW)
        FAH2_TML_hydr_electr_power_kW = \
            hydr_electr_elec_kJ_per_mol_FA * FAH2_TML_FA_flow_mol_per_sec
            
    # calculate CO2 electrolyzer energy (kWh/kg H2)
    FAH2_TML_hydr_electr_elec_kWh_per_kg = \
        FAH2_TML_hydr_electr_power_kW / tot_H2_deliv_kg_per_hr
        
    # convert CO2 electrolyzer energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_TML_hydr_electr_elec_MJ_per_MJ = \
        FAH2_TML_hydr_electr_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg

    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        FAH2_TML_hydr_electr_elec_kWh_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_TML_hydr_electr_elec_MJ_per_MJ
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: CO2 electrolyzer energy emissions
        
    # calculate CO2 electrolyzer energy emissions (kg CO2/kg H2)
    FAH2_TML_hydr_electr_ghg_kg_CO2_per_kg = \
        FAH2_TML_hydr_electr_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert CO2 electrolyzer energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_TML_hydr_electr_ghg_g_CO2_per_MJ = \
        FAH2_TML_hydr_electr_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_TML_hydr_electr_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_TML_hydr_electr_ghg_g_CO2_per_MJ
        ])

    # ------------------------------------------------------------------------
    # production - formic acid: CO2 electrolyzer energy cost
    
    # calculate CO2 electrolyzer energy cost ($/kg H2)
    FAH2_TML_hydr_electr_elec_cost_usd_per_kg = \
        FAH2_TML_hydr_electr_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate CO2 electrolyzer energy cost ($/yr)
    FAH2_TML_hydr_electr_elec_cost_usd_per_yr = \
        FAH2_TML_hydr_electr_elec_cost_usd_per_kg * \
        tot_H2_deliv_kg_per_yr
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        FAH2_TML_hydr_electr_elec_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        FAH2_TML_hydr_electr_elec_cost_usd_per_kg
        ])
        
    # ------------------------------------------------------------------------
    # production - formic acid: 
    # CO2 electrolyzer installed cost and annual O&M cost
        
    # initialize CO2 electrolyzer installed cost and 
    # annual O&M cost (zero by default)
    FAH2_TML_hydr_electr_inst_cost_usd = 0.0
    FAH2_TML_hydr_electr_om_cost_usd_per_yr = 0.0
    FAH2_TML_hydr_electr_dollar_year = output_dollar_year

    if FA_prod_pathway == 'electro':      
        
        # TODO: turn electrolyzer area calculations into function 
        # ideal assumptions for now

        # calculate electrolyzer area (m^2) required
        FAH2_TML_hydr_electr_area_sq_m = \
            e_per_mol_FA * faraday_const_C_per_mol_e / \
            electr_curr_dens_A_per_sq_m * FAH2_TML_FA_flow_mol_per_sec
                    
        # calculate CO2 electrolyzer installed cost ($) and annual 
        # O&M cost ($/yr), both in output dollar year
        FAH2_TML_hydr_electr_inst_cost_usd, \
        FAH2_TML_hydr_electr_om_cost_usd_per_yr, \
        FAH2_TML_hydr_electr_dollar_year = \
            electrolyzer_fixed_costs(
                    electr_area_sq_m = FAH2_TML_hydr_electr_area_sq_m, 
                    output_dollar_year = output_dollar_year,
                    electr_purc_cost_usd_per_sq_m = \
                        electr_purc_cost_usd_per_sq_m
                    )
            
    # calculate CO2 electrolyzer O&M cost ($/kg H2)
    FAH2_TML_hydr_electr_om_cost_usd_per_kg = \
        FAH2_TML_hydr_electr_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_TML_hydr_electr_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_TML_hydr_electr_om_cost_usd_per_kg
        ])    
    
    # ------------------------------------------------------------------------
    # production - formic acid: 
    # formic acid purification (distillation) energy consumption
       
    # TODO: add water / formic acid distillation energy consumption
    # get from Components modeling?
    
    # ------------------------------------------------------------------------
    # production - formic acid: 
    # formic acid purification (distillation) energy emissions

    # TODO: add water / formic acid distillation energy emissions

    # ------------------------------------------------------------------------
    # production - formic acid: 
    # formic acid purification (distillation) energy cost
    
    # TODO: add water / formic acid distillation energy cost
    
    # ------------------------------------------------------------------------
    # production - formic acid: 
    # formic acid purification (distillation) installed cost 
    # and annual O&M cost

    # TODO: add water / formic acid distillation installed cost and
    # annual O&M cost
    # get equipment size from Components modeling?
    
    # initialize formic acid purification capital and O&M costs
    # ($ and $/yr, zero by default)
    FAH2_TML_distil_cap_cost_usd = 0.0
    FAH2_TML_distil_om_cost_usd_per_yr = 0.0    
    FAH2_TML_distil_dollar_year = output_dollar_year
        
    # initialize formic acid purification capital and O&M costs 
    # ($/kg FA, zero by default)
    FAH2_TML_distil_lev_cap_cost_usd_per_kgFA = 0.0    
    FAH2_TML_distil_om_cost_usd_per_kgFA = 0.0
    
    # if produce formic acid at terminal (as opposed to purchase), 
    # update formic acid purification costs ($/kg FA)
    # placeholders from Table 8, Ramdin et al., 2019; 
    # hybrid extraction-distillation from 10 wt% to 85 wt% FA    
    # "Maintenance, depreciation, interest, and taxes are 
    # excluded" in OPEX in Ramdin et al."
    # TODO: revisit: apply to both thermo and electro production?
    # TODO: calculate capital cost ($) and O&M cost ($/yr) as function of
    # plant capacity
    if FA_prod_pathway != 'purchase':
        FAH2_TML_distil_lev_cap_cost_usd_per_kgFA = 0.129
        FAH2_TML_distil_om_cost_usd_per_kgFA = 0.116    
        
    # calculate formic acid purification capital cost ($/kg H2)
    FAH2_TML_distil_lev_cap_cost_usd_per_kg = \
        FAH2_TML_distil_lev_cap_cost_usd_per_kgFA * \
        tot_FA_deliv_kg_per_day / tot_H2_deliv_kg_per_day
        
    # calculate formic acid purification O&M cost ($/kg H2)
    FAH2_TML_distil_om_cost_usd_per_kg = \
        FAH2_TML_distil_om_cost_usd_per_kgFA * \
        tot_FA_deliv_kg_per_day / tot_H2_deliv_kg_per_day

    # append results to list
    # TODO: add formic acid purification O&M cost in $/yr --> 
    # calculate as function of plant capacity
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'separation', 
        'distillation', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_distil_lev_cap_cost_usd_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'separation', 
        'distillation', 
        'O&M cost', 
        'operation cost', 
        '$/kg H2', 
        FAH2_TML_distil_om_cost_usd_per_kg
        ])
    
    #%% CALCULATIONS: LOHC / FORMIC ACID DELIVERY ("FAH2")
    # FORMIC ACID PRECONDITIONING @ TERMINAL
    
    # TODO: update formic acid pumping costs

    # ------------------------------------------------------------------------
    # preconditioning - formic acid: 
    # low-head truck loading pump energy consumption and size
    
    # TODO: revisit flowrate through pump - incoroporate fill time, peak 
    # demand, outage, etc.; for now, use formic acid flowrate through terminal
    
    # # calculate loading pump inlet pressure (bar)
    # FAH2_TML_in_pres_bar = \
    #     FAH2_TML_in_pres_atm * Pa_per_atm / Pa_per_bar
    
    # # calculate loading pump outlet pressure (bar)
    # FAH2_TML_out_pres_bar = \
    #     FAH2_TML_out_pres_atm * Pa_per_atm / Pa_per_bar
    
    # # calculate loading pump power (kW) and size (kW/pump) at terminal
    # FAH2_TML_load_pump_tot_power_kW, \
    # FAH2_TML_load_pump_power_kW_per_pump, \
    # FAH2_TML_num_pumps = \
    #     pump_power_and_size(
    #         out_pres_bar = FAH2_TML_out_pres_bar,
    #         in_pres_bar = FAH2_TML_in_pres_bar,
    #         fluid_flow_kg_per_sec = TML_H2_flow_kg_per_sec, 
    #         dens_kg_per_cu_m = dens_liq_H2_kg_per_cu_m
    #         )
    
    # # calculate loading pump energy (kWh/kg H2)
    # FAH2_TML_load_pump_elec_kWh_per_kg = \
    #     FAH2_TML_load_pump_tot_power_kW / tot_H2_deliv_kg_per_hr
    
    # # convert loading pump energy to MJ/MJ H2 (LHV) 
    # # (for comparison with HDSAM V3.1)
    # FAH2_TML_load_pump_elec_MJ_per_MJ = \
    #     FAH2_TML_load_pump_elec_kWh_per_kg * MJ_per_kWh / \
    #     low_heat_val_H2_MJ_per_kg
    
    FAH2_TML_load_pump_elec_kWh_per_kg = 0.0
    FAH2_TML_load_pump_elec_MJ_per_MJ = 0.0

    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        FAH2_TML_load_pump_elec_kWh_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_TML_load_pump_elec_MJ_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # preconditioning - formic acid: 
    # low-head truck loading pump energy emissions

    # # calculate loading pump energy emissions (kg CO2/kg H2)
    # FAH2_TML_load_pump_ghg_kg_CO2_per_kg = \
    #     FAH2_TML_load_pump_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # # convert loading pump energy emissions to g CO2/MJ H2 (LHV) 
    # # (for comparison with HDSAM V3.1)
    # FAH2_TML_load_pump_ghg_g_CO2_per_MJ = \
    #     FAH2_TML_load_pump_ghg_kg_CO2_per_kg * g_per_kg / \
    #     low_heat_val_H2_MJ_per_kg
    
    FAH2_TML_load_pump_ghg_kg_CO2_per_kg = 0.0
    FAH2_TML_load_pump_ghg_g_CO2_per_MJ = 0.0
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_TML_load_pump_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_TML_load_pump_ghg_g_CO2_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # preconditioning - formic acid: 
    # low-head truck loading pump energy cost
    
    # # calculate loading pump energy cost ($/kg H2)
    # FAH2_TML_load_pump_elec_cost_usd_per_kg = \
    #     FAH2_TML_load_pump_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # # calculate loading pump energy cost ($/yr)
    # FAH2_TML_load_pump_elec_cost_usd_per_yr = \
    #     FAH2_TML_load_pump_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    FAH2_TML_load_pump_elec_cost_usd_per_yr = 0.0
    FAH2_TML_load_pump_elec_cost_usd_per_kg = 0.0
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        FAH2_TML_load_pump_elec_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        FAH2_TML_load_pump_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - formic acid: 
    # low-head truck loading pump installed cost and annual O&M cost
            
    # # calculate formic acid volumetric flowrate through 
    # # loading pump (m^3/hr)
    # FAH2_TML_FA_flow_cu_m_per_hr = \
    #     FAH2_TML_FA_flow_kg_per_day / dens_FA_kg_per_cu_m / hr_per_day

    # # calculate loading pump installed cost ($) and annual 
    # # O&M cost ($/yr), both in output dollar year
    # FAH2_TML_load_pump_inst_cost_usd, \
    # FAH2_TML_load_pump_om_cost_usd_per_yr, \
    # FAH2_TML_load_pump_dollar_year = \
    #     low_head_pump_fixed_costs(
    #         num_pumps = FAH2_TML_num_pumps, 
    #         fluid_flow_cu_m_per_hr = FAH2_TML_H2_flow_cu_m_per_hr,
    #         output_dollar_year = output_dollar_year
    #         )
    
    # # calculate loading pump O&M cost ($/kg H2)
    # FAH2_TML_load_pump_om_cost_usd_per_kg = \
    #     FAH2_TML_load_pump_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    FAH2_TML_load_pump_inst_cost_usd = 0.0
    FAH2_TML_load_pump_dollar_year = output_dollar_year
    FAH2_TML_load_pump_om_cost_usd_per_yr = 0.0
    FAH2_TML_load_pump_om_cost_usd_per_kg = 0.0
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_TML_load_pump_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_TML_load_pump_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # preconditioning - formic acid: 
    # terminal formic acid storage installed cost and annual O&M cost
            
    # calculate total formic acid storage capacity (m^3) and 
    # number of storage tanks required at terminal
    FAH2_TML_stor_tank_capacity_cu_m, \
    FAH2_TML_num_tanks = \
        general_tank_storage_size(
            fluid_flow_kg_per_day = FAH2_TML_FA_flow_kg_per_day,
            stor_amt_days = FAH2_TML_stor_amt_days,
            fluid_dens_kg_per_cu_m = dens_FA_kg_per_cu_m
            )
    
    # calculate storage installed cost ($) and annual O&M cost ($/yr), 
    # both in output dollar year    
    FAH2_TML_stor_inst_cost_usd, \
    FAH2_TML_stor_om_cost_usd_per_yr, \
    FAH2_TML_stor_dollar_year = \
        general_tank_stor_fixed_costs(
            stor_tank_capacity_cu_m = FAH2_TML_stor_tank_capacity_cu_m,
            num_tanks = FAH2_TML_num_tanks,
            output_dollar_year = output_dollar_year, 
            material = 'fiber glass open top'
            )

    # calculate storage O&M cost ($/kg H2)
    FAH2_TML_stor_om_cost_usd_per_kg = \
        FAH2_TML_stor_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_TML_stor_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_TML_stor_om_cost_usd_per_kg
        ])    
    
    #%% CALCULATIONS: LOHC / FORMIC ACID DELIVERY ("FAH2")
    # FORMIC ACID PRODUCTION + PRECONDITIONING @ TERMINAL

    # ------------------------------------------------------------------------
    # production + preconditioning - formic acid: 
    # terminal total capital investment and "other" annual O&M costs

    # "Other" O&M costs include insurance, property taxes, licensing and 
    # permits. Operation, maintenance, and repair costs for individual 
    # refueling station components (e.g., compressor) are calculated 
    # separately.
        
    # calculate terminal total initial capital investment ($)
    # (= hydrogenation hydrogen compressor, reactor, electrolyzer, 
    # separator, etc. installed costs)
    FAH2_TML_init_cap_inv_usd = \
        FAH2_TML_hydr_compr_inst_cost_usd + \
        FAH2_TML_hydr_pump_inst_cost_usd + \
        FAH2_TML_hydr_vap_inst_cost_usd + \
        FAH2_TML_hydr_react_inst_cost_usd + \
        FAH2_TML_hydr_catal_purc_cost_usd + \
        FAH2_TML_hydr_electr_inst_cost_usd + \
        FAH2_TML_distil_cap_cost_usd + \
        FAH2_TML_load_pump_inst_cost_usd + \
        FAH2_TML_stor_inst_cost_usd
                
    # calculate terminal cost allocations (%) to hydrogenation hydrogen
    # compressor, reactor, electrolyzer, separator, etc.
    # % of terminal total initial capital investment
    # use to allocate total capital investment, other O&M costs, and labor 
    # cost
    
    if FAH2_TML_init_cap_inv_usd == 0.0:
        FAH2_TML_hydr_compr_cost_perc = 0.0
        FAH2_TML_hydr_pump_cost_perc = 0.0
        FAH2_TML_hydr_vap_cost_perc = 0.0
        FAH2_TML_hydr_react_cost_perc = 0.0
        FAH2_TML_hydr_catal_cost_perc = 0.0
        FAH2_TML_hydr_electr_cost_perc = 0.0
        FAH2_TML_distil_cost_perc = 0.0
        FAH2_TML_load_pump_cost_perc = 0.0
        FAH2_TML_stor_cost_perc = 0.0
    else:
        FAH2_TML_hydr_compr_cost_perc = \
            FAH2_TML_hydr_compr_inst_cost_usd / \
            FAH2_TML_init_cap_inv_usd
        FAH2_TML_hydr_pump_cost_perc = \
            FAH2_TML_hydr_pump_inst_cost_usd / \
            FAH2_TML_init_cap_inv_usd
        FAH2_TML_hydr_vap_cost_perc = \
            FAH2_TML_hydr_vap_inst_cost_usd / \
            FAH2_TML_init_cap_inv_usd
        FAH2_TML_hydr_react_cost_perc = \
            FAH2_TML_hydr_react_inst_cost_usd / \
            FAH2_TML_init_cap_inv_usd
        FAH2_TML_hydr_catal_cost_perc = \
            FAH2_TML_hydr_catal_purc_cost_usd / \
            FAH2_TML_init_cap_inv_usd
        FAH2_TML_hydr_electr_cost_perc = \
            FAH2_TML_hydr_electr_inst_cost_usd / \
            FAH2_TML_init_cap_inv_usd
        FAH2_TML_distil_cost_perc = \
            FAH2_TML_distil_cap_cost_usd / \
            FAH2_TML_init_cap_inv_usd
        FAH2_TML_load_pump_cost_perc = \
            FAH2_TML_load_pump_inst_cost_usd / \
            FAH2_TML_init_cap_inv_usd
        FAH2_TML_stor_cost_perc = \
            FAH2_TML_stor_inst_cost_usd / \
            FAH2_TML_init_cap_inv_usd

    # check whether cost allocations (%) sum to one
    # raise error if false
    if FAH2_TML_init_cap_inv_usd == 0.0:
        pass
    elif abs(
            FAH2_TML_hydr_compr_cost_perc + \
            FAH2_TML_hydr_pump_cost_perc + \
            FAH2_TML_hydr_vap_cost_perc + \
            FAH2_TML_hydr_react_cost_perc + \
            FAH2_TML_hydr_catal_cost_perc + \
            FAH2_TML_hydr_electr_cost_perc + \
            FAH2_TML_distil_cost_perc + \
            FAH2_TML_load_pump_cost_perc + \
            FAH2_TML_stor_cost_perc - \
            1.0
            ) >= 1.0e-9:
        raise ValueError(
            'Component cost allocations need to sum to one.'
            )
        
    # check if all terminal components have the same dollar year
    # if true, assign dollar year of refueling station costs to the dollar 
    # year of one of the components 
    if (FAH2_TML_hydr_compr_dollar_year == \
        FAH2_TML_hydr_pump_dollar_year) \
        and (FAH2_TML_hydr_pump_dollar_year == \
             FAH2_TML_hydr_vap_dollar_year) \
        and (FAH2_TML_hydr_vap_dollar_year == \
             FAH2_TML_hydr_react_dollar_year) \
        and (FAH2_TML_hydr_react_dollar_year == \
             FAH2_TML_hydr_electr_dollar_year) \
        and (FAH2_TML_hydr_electr_dollar_year == \
             FAH2_TML_distil_dollar_year) \
        and (FAH2_TML_distil_dollar_year == \
             FAH2_TML_load_pump_dollar_year) \
        and (FAH2_TML_load_pump_dollar_year == \
             FAH2_TML_stor_dollar_year):
        FAH2_TML_dollar_year = FAH2_TML_hydr_compr_dollar_year
    else:
        raise ValueError(
            'Dollar year of components need to match.'
            )

    # calculate terminal total capital investment ($, output dollar year) 
    FAH2_TML_tot_cap_inv_usd, \
    FAH2_TML_cap_cost_dollar_year = \
        non_station_total_capital_investment(
            init_cap_inv_usd = FAH2_TML_init_cap_inv_usd, 
            input_dollar_year = FAH2_TML_dollar_year
            )
    
    # calculate terminal "other" annual O&M costs ($/yr, output dollar year)
    FAH2_TML_om_cost_usd_per_yr, \
    FAH2_TML_om_cost_dollar_year = \
        other_om_cost(
            tot_cap_inv_usd = FAH2_TML_tot_cap_inv_usd,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate terminal "other" O&M costs ($/kg H2)
    FAH2_TML_om_cost_usd_per_kg = \
        FAH2_TML_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign "other" O&M costs to hydrogenation hydrogen compressor, 
    # reactor, electrolyzer, separator, etc.
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_hydr_compr_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_hydr_compr_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_hydr_pump_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_hydr_pump_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'vaporization', 
        'reactor vaporizer', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_hydr_vap_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'vaporization', 
        'reactor vaporizer', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_hydr_vap_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_hydr_react_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_hydr_react_cost_perc
        ])
                
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'catalyst', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_hydr_catal_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'catalyst', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_hydr_catal_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_hydr_electr_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_hydr_electr_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'separation', 
        'distillation', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_distil_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'separation', 
        'distillation', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_distil_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_load_pump_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_load_pump_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_TML_om_cost_usd_per_yr * \
            FAH2_TML_stor_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_TML_om_cost_usd_per_kg * \
            FAH2_TML_stor_cost_perc
        ])
                
    # ------------------------------------------------------------------------
    # production + preconditioning - formic acid: 
    # terminal annual labor cost 

    # TODO: revisit labor cost scaling for formic acid terminal

    # calculate terminal annual labor cost ($/yr, output dollar year), 
    # including overhead and G&A
    FAH2_TML_labor_cost_usd_per_yr, \
    FAH2_TML_labor_cost_dollar_year = \
        non_station_labor_cost(
            H2_flow_kg_per_day = FAH2_TML_H2_flow_kg_per_day, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate terminal labor cost ($/kg H2)
    FAH2_TML_labor_cost_usd_per_kg = \
        FAH2_TML_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign labor cost to hydrogenation hydrogen compressor, 
    # reactor, electrolyzer, separator, etc.
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_hydr_compr_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_hydr_compr_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_hydr_pump_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_hydr_pump_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'vaporization', 
        'reactor vaporizer', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_hydr_vap_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'vaporization', 
        'reactor vaporizer', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_hydr_vap_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_hydr_react_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_hydr_react_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'catalyst', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_hydr_catal_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'catalyst', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_hydr_catal_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_hydr_electr_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_hydr_electr_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'separation', 
        'distillation', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_distil_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'separation', 
        'distillation', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_distil_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_load_pump_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_load_pump_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_TML_labor_cost_usd_per_yr * \
            FAH2_TML_stor_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_TML_labor_cost_usd_per_kg * \
            FAH2_TML_stor_cost_perc
        ])
        
    # ------------------------------------------------------------------------
    # production - formic acid:
    # hydrogenation hydrogen compressor levelized capital cost
    
    # calculate hydrogenation hydrogen compressor total capital investment ($) 
    # (= terminal total capital investment allocated to compressor)
    FAH2_TML_hydr_compr_tot_cap_inv_usd = \
        FAH2_TML_hydr_compr_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate hydrogenation hydrogen compressor levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_hydr_compr_lev_cap_cost_usd_per_yr, \
    FAH2_TML_hydr_compr_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_hydr_compr_tot_cap_inv_usd, 
            life_yr = TML_compr_life_yr, 
            depr_yr = TML_compr_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate hydrogenation hydrogen compressor levelized 
    # capital cost ($/kg H2)
    FAH2_TML_hydr_compr_lev_cap_cost_usd_per_kg = \
        FAH2_TML_hydr_compr_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_TML_hydr_compr_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'compression', 
        'reactor compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_hydr_compr_lev_cap_cost_usd_per_kg
        ])
            
    # ------------------------------------------------------------------------
    # production - formic acid:
    # hydrogenation CO2 pump levelized capital cost
    
    # calculate hydrogenation CO2 pump total capital investment ($) 
    # (= terminal total capital investment allocated to pump)
    FAH2_TML_hydr_pump_tot_cap_inv_usd = \
        FAH2_TML_hydr_pump_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate hydrogenation CO2 pump levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_hydr_pump_lev_cap_cost_usd_per_yr, \
    FAH2_TML_hydr_pump_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_hydr_pump_tot_cap_inv_usd, 
            life_yr = TML_pump_life_yr, 
            depr_yr = TML_pump_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate hydrogenation CO2 pump levelized 
    # capital cost ($/kg H2)
    FAH2_TML_hydr_pump_lev_cap_cost_usd_per_kg = \
        FAH2_TML_hydr_pump_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'pumping', 
        'reactor pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_TML_hydr_pump_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'pumping', 
        'reactor pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_hydr_pump_lev_cap_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # production - formic acid:
    # hydrogenation CO2 vaporizer levelized capital cost
    
    # calculate hydrogenation CO2 vaporizer total capital investment ($) 
    # (= terminal total capital investment allocated to vaporizer)
    FAH2_TML_hydr_vap_tot_cap_inv_usd = \
        FAH2_TML_hydr_vap_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate hydrogenation CO2 vaporizer levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_hydr_vap_lev_cap_cost_usd_per_yr, \
    FAH2_TML_hydr_vap_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_hydr_vap_tot_cap_inv_usd, 
            life_yr = TML_vap_life_yr, 
            depr_yr = TML_vap_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate hydrogenation CO2 vaporizer levelized 
    # capital cost ($/kg H2)
    FAH2_TML_hydr_vap_lev_cap_cost_usd_per_kg = \
        FAH2_TML_hydr_vap_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'vaporization', 
        'reactor vaporizer', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_TML_hydr_vap_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'vaporization', 
        'reactor vaporizer', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_hydr_vap_lev_cap_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # production - formic acid:
    # hydrogenation reactor levelized capital cost
    
    # calculate hydrogenation reactor total capital investment ($) 
    # (= terminal total capital investment allocated to reactor)
    FAH2_TML_hydr_react_tot_cap_inv_usd = \
        FAH2_TML_hydr_react_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate hydrogenation reactor levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_hydr_react_lev_cap_cost_usd_per_yr, \
    FAH2_TML_hydr_react_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_hydr_react_tot_cap_inv_usd, 
            life_yr = TML_react_life_yr, 
            depr_yr = TML_react_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate hydrogenation reactor levelized 
    # capital cost ($/kg H2)
    FAH2_TML_hydr_react_lev_cap_cost_usd_per_kg = \
        FAH2_TML_hydr_react_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'reactor', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_TML_hydr_react_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'reactor', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_hydr_react_lev_cap_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # production - formic acid:
    # hydrogenation catalyst levelized capital cost
    
    # calculate hydrogenation catalyst total capital investment ($) 
    # (= terminal total capital investment allocated to catalyst)
    FAH2_TML_hydr_catal_tot_cap_inv_usd = \
        FAH2_TML_hydr_catal_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate hydrogenation catalyst levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_hydr_catal_lev_cap_cost_usd_per_yr, \
    FAH2_TML_hydr_catal_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_hydr_catal_tot_cap_inv_usd, 
            life_yr = FAH2_hydr_catal_life_yr, 
            depr_yr = FAH2_hydr_catal_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate hydrogenation catalyst levelized 
    # capital cost ($/kg H2)
    FAH2_TML_hydr_catal_lev_cap_cost_usd_per_kg = \
        FAH2_TML_hydr_catal_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'catalyst', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_TML_hydr_catal_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'catalyst', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_hydr_catal_lev_cap_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # production - formic acid:
    # CO2 electrolyzer levelized capital cost
    
    # calculate CO2 electrolyzer total capital investment ($) 
    # (= terminal total capital investment allocated to electrolyzer)
    FAH2_TML_hydr_electr_tot_cap_inv_usd = \
        FAH2_TML_hydr_electr_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate CO2 electrolyzer levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_hydr_electr_lev_cap_cost_usd_per_yr, \
    FAH2_TML_hydr_electr_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_hydr_electr_tot_cap_inv_usd, 
            life_yr = TML_electr_life_yr, 
            depr_yr = TML_electr_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate CO2 electrolyzer levelized capital cost ($/kg H2)
    FAH2_TML_hydr_electr_lev_cap_cost_usd_per_kg = \
        FAH2_TML_hydr_electr_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_TML_hydr_electr_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'terminal', 
        'reaction', 
        'CO2 electrolyzer', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_hydr_electr_lev_cap_cost_usd_per_kg
        ])    
    
    # ------------------------------------------------------------------------
    # production - formic acid:
    # formic acid purification (distillation) levelized capital cost
    
    # calculate distillator total capital investment ($) 
    # (= terminal total capital investment allocated to distillator)
    FAH2_TML_distil_tot_cap_inv_usd = \
        FAH2_TML_distil_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate distillator levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_distil_lev_cap_cost_usd_per_yr, \
    FAH2_TML_distil_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_distil_tot_cap_inv_usd, 
            life_yr = TML_distil_life_yr, 
            depr_yr = TML_distil_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate distillator levelized capital cost ($/kg H2)
    # TODO: uncomment after updating distillator capital cost ($) 
    # FAH2_TML_distil_lev_cap_cost_usd_per_kg = \
    #     FAH2_TML_distil_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # TODO: uncomment after updating distillator capital cost ($) 
    # (for now, use $/kg values from Ramdin et al., 2019)
    # list_output.append([
    #     'LOHC - formic acid', 
    #     'production', 
    #     'terminal', 
    #     'separation', 
    #     'distillation', 
    #     'capital cost', 
    #     'levelized capital cost', 
    #     '$/yr', 
    #     FAH2_TML_distil_lev_cap_cost_usd_per_yr
    #     ])
    # list_output.append([
    #     'LOHC - formic acid', 
    #     'production', 
    #     'terminal', 
    #     'separation', 
    #     'distillation', 
    #     'capital cost', 
    #     'levelized capital cost', 
    #     '$/kg H2', 
    #     FAH2_TML_distil_lev_cap_cost_usd_per_kg
    #     ])

    # ------------------------------------------------------------------------
    # preconditioning - formic acid:
    # loading pump levelized capital cost
    
    # calculate loading pump total capital investment ($) 
    # (= terminal total capital investment allocated to loading pump)
    FAH2_TML_load_pump_tot_cap_inv_usd = \
        FAH2_TML_load_pump_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate loading pump levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_load_pump_lev_cap_cost_usd_per_yr, \
    FAH2_TML_load_pump_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_load_pump_tot_cap_inv_usd, 
            life_yr = TML_pump_life_yr, 
            depr_yr = TML_pump_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate loading pump levelized capital cost ($/kg H2)
    FAH2_TML_load_pump_lev_cap_cost_usd_per_kg = \
        FAH2_TML_load_pump_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_TML_load_pump_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'pumping', 
        'loading pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_load_pump_lev_cap_cost_usd_per_kg
        ])    

    # ------------------------------------------------------------------------
    # preconditioning - formic acid:
    # terminal formic acid storage levelized capital cost
    
    # calculate terminal formic acid storage total capital investment ($) 
    # (= terminal total capital investment allocated to storage)
    FAH2_TML_stor_tot_cap_inv_usd = \
        FAH2_TML_stor_cost_perc * FAH2_TML_tot_cap_inv_usd
    
    # calculate terminal formic acid storage levelized capital cost 
    # ($/yr, output dollar year)
    FAH2_TML_stor_lev_cap_cost_usd_per_yr, \
    FAH2_TML_stor_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_TML_stor_tot_cap_inv_usd, 
            life_yr = TML_stor_life_yr, 
            depr_yr = TML_stor_depr_yr,
            input_dollar_year = FAH2_TML_cap_cost_dollar_year
            )
    
    # calculate terminal formic acid storage levelized capital cost ($/kg H2)
    FAH2_TML_stor_lev_cap_cost_usd_per_kg = \
        FAH2_TML_stor_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'liquid storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_TML_stor_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'preconditioning', 
        'terminal', 
        'storage', 
        'liquid storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_TML_stor_lev_cap_cost_usd_per_kg
        ])    

    #%% CALCULATIONS: LOHC / FORMIC ACID DELIVERY ("FAH2")
    # LOHC / FORMIC ACID TRANSPORT @ TRUCK

    # ------------------------------------------------------------------------
    # transport - formic acid: 
    # truck fuel consumption, number of trucks required, number of deliveries 
    # per day, total trip time
        
    # calculate truck fuel consumption (gallon/kg formic acid), number of 
    # trucks, number of deliveries per day, total trip time (hr/trip)
    # TODO: revisit truck loading and unloading times 
    # (for now, use same truck loading and unloading times as liquid hydrogen 
    # trucks)
    FAH2_truck_fuel_gal_per_kgFA, \
    FAH2_num_trucks, \
    FAH2_truck_num_delivs_per_day, \
    FAH2_truck_trip_time_hr = \
        transport_energy(
            deliv_dist_mi = deliv_dist_mi_ow, 
            speed_mi_per_hr = truck_speed_mi_per_hr, 
            load_unload_time_hr = liq_truck_load_unload_time_hr,
            fuel_econ_mi_per_gal = truck_fuel_econ_mi_per_gal,
            deliv_capacity_kg = liq_truck_deliv_capacity_kgFA, 
            cargo_flow_kg_per_day = FAH2_TML_FA_flow_kg_per_day
            )
    
    # calculate truck fuel consumption (gallon/kg H2)
    FAH2_truck_fuel_gal_per_kg = \
        FAH2_truck_fuel_gal_per_kgFA * \
        FAH2_TML_FA_flow_kg_per_day / tot_H2_deliv_kg_per_day
    
    # convert truck fuel consumption to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_truck_fuel_MJ_per_MJ = \
        FAH2_truck_fuel_gal_per_kg * low_heat_val_diesel_MJ_per_gal / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy consumption', 
        'diesel consumption', 
        'gallon/kg H2', 
        FAH2_truck_fuel_gal_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy consumption', 
        'diesel consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_truck_fuel_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # transport - formic acid: 
    # truck fuel emissions

    # calculate truck fuel emissions (kg CO2/kg H2)
    FAH2_truck_ghg_kg_CO2_per_kg = \
        FAH2_truck_fuel_gal_per_kg * diesel_ghg_kg_CO2_per_gal
    
    # convert truck fuel emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_truck_ghg_g_CO2_per_MJ = \
        FAH2_truck_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_truck_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_truck_ghg_g_CO2_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # transport - formic acid: truck fuel cost
    
    # calculate truck fuel cost ($/kg H2)
    FAH2_truck_fuel_cost_usd_per_kg = \
        FAH2_truck_fuel_gal_per_kg * diesel_cost_usd_per_gal
    
    # calculate truck fuel cost ($/yr)
    FAH2_truck_fuel_cost_usd_per_yr = \
        FAH2_truck_fuel_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy cost', 
        'fuel cost', 
        '$/yr', 
        FAH2_truck_fuel_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'energy cost', 
        'fuel cost', 
        '$/kg H2', 
        FAH2_truck_fuel_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # transport - formic acid: truck capital cost and annual O&M cost
    
    # calculate number of deliveries (truck-trips) per year for formic acid
    FAH2_truck_num_delivs_per_yr = \
        FAH2_truck_num_delivs_per_day * day_per_yr 
            
    # calculate formic acid truck capital cost ($, output dollar year) 
    FAH2_truck_cap_cost_usd, \
    FAH2_truck_cap_cost_dollar_year = \
        liquid_truck_capital_cost(
            num_trucks = FAH2_num_trucks,
            output_dollar_year = output_dollar_year
            )
    
    # calculate truck annual O&M cost ($/yr, output dollar year)
    FAH2_truck_om_cost_usd_per_yr, \
    FAH2_truck_om_cost_dollar_year = \
        truck_om_cost(
            deliv_dist_mi = deliv_dist_mi_ow, 
            num_delivs_per_yr = FAH2_truck_num_delivs_per_yr, 
            output_dollar_year = output_dollar_year
            )
        
    # calculate truck O&M cost ($/kg H2)
    FAH2_truck_om_cost_usd_per_kg = \
        FAH2_truck_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_truck_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_truck_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # transport - formic acid: truck annual labor cost
    
    # calculate truck annual labor cost ($/yr, output dollar year), 
    # including overhead and G&A
    FAH2_truck_labor_cost_usd_per_yr, \
    FAH2_truck_labor_cost_dollar_year = \
        truck_labor_cost(
            num_delivs_per_yr = FAH2_truck_num_delivs_per_yr, 
            trip_time_hr = FAH2_truck_trip_time_hr, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate truck labor cost ($/kg H2)
    FAH2_truck_labor_cost_usd_per_kg = \
        FAH2_truck_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_truck_labor_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_truck_labor_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # transport - formic acid: truck levelized capital cost
    
    # calculate truck levelized capital cost ($/yr, output dollar year)
    # truck total capital investment = truck capital cost
    FAH2_truck_lev_cap_cost_usd_per_yr, \
    FAH2_truck_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_truck_cap_cost_usd, 
            life_yr = truck_life_yr, 
            depr_yr = truck_depr_yr,
            input_dollar_year = FAH2_truck_cap_cost_dollar_year
            )
    
    # calculate truck levelized capital cost ($/kg H2)
    FAH2_truck_lev_cap_cost_usd_per_kg = \
        FAH2_truck_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_truck_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'transport', 
        'truck', 
        'trucking', 
        'truck', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_truck_lev_cap_cost_usd_per_kg
        ])

    #%% CALCULATIONS: LOHC / FORMIC ACID DELIVERY ("FAH2")
    # CO2 TRANSPORT @ TRUCK

    # ------------------------------------------------------------------------
    # production - formic acid: CO2 recycling (for hydrogenation) costs
    
    # initialize CO2 recycling costs (zero by default)
    FAH2_CO2_recyc_cost_usd_per_tCO2 = 0.0
    FAH2_CO2_recyc_cost_usd_per_yr_per_stn = 0.0
    FAH2_CO2_recyc_cost_dollar_year = output_dollar_year
    
    # if produce formic acid at terminal (as opposed to purchase),
    # calculate CO2 recycling cost
    if FA_prod_pathway != 'purchase':
        
        # calculate CO2 recycling cost per station
        # ($/tonne CO2 and $/yr, output dollar year)
        FAH2_CO2_recyc_cost_usd_per_tCO2, \
        FAH2_CO2_recyc_cost_usd_per_yr_per_stn, \
        FAH2_CO2_recyc_cost_dollar_year = \
            CO2_transport_all_in_cost(
                CO2_flow_kt_per_yr = FAH2_STN_CO2_flow_kt_per_yr,
                deliv_dist_mi = deliv_dist_mi_ow,
                output_dollar_year = output_dollar_year
                )        
        
    # calculate CO2 recycling cost ($/yr)
    # sum of all stations, excluding LOHC truck capital cost (shared)
    FAH2_CO2_recyc_cost_usd_per_yr = max(
        0, 
        FAH2_CO2_recyc_cost_usd_per_yr_per_stn * target_num_stns - \
        FAH2_truck_lev_cap_cost_usd_per_yr
        )

    # calculate CO2 recycling cost ($/kg H2)
    FAH2_CO2_recyc_cost_usd_per_kg = \
        FAH2_CO2_recyc_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
        
    # append results to list
    # TODO: allocate "all-in" cost to capital, labor, fuel, etc.
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'truck', 
        'CO2 recycling', 
        'truck', 
        'capital cost', 
        'all-in CO2 transport cost', 
        '$/yr', 
        FAH2_CO2_recyc_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'truck', 
        'CO2 recycling', 
        'truck', 
        'capital cost', 
        'all-in CO2 transport cost', 
        '$/kg H2', 
        FAH2_CO2_recyc_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # production - formic acid: CO2 recycling (for hydrogenation) emissions

    # TODO: add CO2 recycling emissions
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'truck', 
        'CO2 recycling', 
        'truck', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        0.0
        ])
    list_output.append([
        'LOHC - formic acid', 
        'production', 
        'truck', 
        'CO2 recycling', 
        'truck', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        0.0
        ])

            
    #%% CALCULATIONS: LOHC / FORMIC ACID DELIVERY ("FAH2")
    # RECONDITIONING @ REFUELING STATION

    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation pump energy consumption and size
    
    # initialize dehydrogenation pump power (kW) and size (kW/pump)
    # (zero by default)
    FAH2_STN_dehydr_pump_tot_power_kW = 0.0
    FAH2_STN_dehydr_pump_power_kW_per_pump = 0.0
    FAH2_STN_num_dehydr_compr = 0

    if FAH2_dehydr_pres_bar > FAH2_STN_dehydr_pump_in_pres_bar:
        
        # calculate dehydrogenation pump power (kW) and size (kW/pump) 
        # at refueling station
        # pump outlet pressure = dehydrogenation reaction pressure
        FAH2_STN_dehydr_pump_tot_power_kW, \
        FAH2_STN_dehydr_pump_power_kW_per_pump, \
        FAH2_STN_num_dehydr_compr = \
            pump_power_and_size(
                out_pres_bar = FAH2_dehydr_pres_bar,
                in_pres_bar = FAH2_STN_dehydr_pump_in_pres_bar,
                fluid_flow_kg_per_sec = FAH2_STN_FA_flow_kg_per_sec, 
                dens_kg_per_cu_m = dens_FA_kg_per_cu_m
                )
            
    # calculate dehydrogenation pump energy (kWh/kg H2)
    FAH2_STN_dehydr_pump_elec_kWh_per_kg = \
        FAH2_STN_dehydr_pump_tot_power_kW * target_num_stns / \
        tot_H2_deliv_kg_per_hr
    
    # convert dehydrogenation pump energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_dehydr_pump_elec_MJ_per_MJ = \
        FAH2_STN_dehydr_pump_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        FAH2_STN_dehydr_pump_elec_kWh_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_STN_dehydr_pump_elec_MJ_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation pump energy emissions
    
    # calculate dehydrogenation pump energy emissions (kg CO2/kg H2)
    FAH2_STN_dehydr_pump_ghg_kg_CO2_per_kg = \
        FAH2_STN_dehydr_pump_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert dehydrogenation pump energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_dehydr_pump_ghg_g_CO2_per_MJ = \
        FAH2_STN_dehydr_pump_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_STN_dehydr_pump_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_STN_dehydr_pump_ghg_g_CO2_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation pump energy cost
    
    # calculate dehydrogenation pump energy cost ($/kg H2)
    FAH2_STN_dehydr_pump_elec_cost_usd_per_kg = \
        FAH2_STN_dehydr_pump_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate dehydrogenation pump energy cost ($/yr)
    FAH2_STN_dehydr_pump_elec_cost_usd_per_yr = \
        FAH2_STN_dehydr_pump_elec_cost_usd_per_kg * \
        tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        FAH2_STN_dehydr_pump_elec_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        FAH2_STN_dehydr_pump_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------        
    # reconditioning - formic acid: 
    # dehydrogenation pump installed cost and annual O&M cost
    
    # initialize dehydrogenation pump installed cost and annual O&M cost 
    # (zero by default)
    FAH2_STN_dehydr_pump_inst_cost_usd_per_stn = 0.0
    FAH2_STN_dehydr_pump_om_cost_usd_per_yr_per_stn = 0.0
    FAH2_STN_dehydr_pump_dollar_year = output_dollar_year
    
    if FAH2_STN_dehydr_pump_tot_power_kW > 0.0:
    
        # calculate formic acid volumetric flowrate through 
        # dehydrogenation pump (m^3/hr)
        FAH2_STN_FA_flow_cu_m_per_hr = \
            FAH2_STN_FA_flow_kg_per_day / dens_FA_kg_per_cu_m / hr_per_day
    
        # calculate dehydrogenation pump installed cost ($) and annual O&M 
        # cost ($/yr) per station, both in output dollar year
        FAH2_STN_dehydr_pump_inst_cost_usd_per_stn, \
        FAH2_STN_dehydr_pump_om_cost_usd_per_yr_per_stn, \
        FAH2_STN_dehydr_pump_dollar_year = \
            low_head_pump_fixed_costs(
                num_pumps = FAH2_STN_num_dehydr_compr, 
                fluid_flow_cu_m_per_hr = FAH2_STN_FA_flow_cu_m_per_hr,
                output_dollar_year = output_dollar_year
                )
    
    # calculate dehydrogenation pump installed cost ($)
    # sum of all stations
    FAH2_STN_dehydr_pump_inst_cost_usd = \
        FAH2_STN_dehydr_pump_inst_cost_usd_per_stn * target_num_stns

    # calculate dehydrogenation pump O&M cost ($/yr)
    # sum of all stations
    FAH2_STN_dehydr_pump_om_cost_usd_per_yr = \
        FAH2_STN_dehydr_pump_om_cost_usd_per_yr_per_stn * \
        target_num_stns

    # calculate dehydrogenation pump O&M cost ($/kg H2)
    FAH2_STN_dehydr_pump_om_cost_usd_per_kg = \
        FAH2_STN_dehydr_pump_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_STN_dehydr_pump_om_cost_usd_per_yr
       ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_STN_dehydr_pump_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation reactor energy consumption

    # TODO: add reactor energy consumption
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation reactor energy emissions

    # TODO: add reactor energy emissions
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation reactor energy cost
    
    # TODO: add reactor energy cost
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation reactor installed cost and annual O&M cost

    # calculate dehydrogenation reactor installed cost ($) and annual 
    # O&M cost ($/yr) per station, both in output dollar year
    FAH2_STN_dehydr_react_inst_cost_usd_per_stn, \
    FAH2_STN_dehydr_react_om_cost_usd_per_yr_per_stn, \
    FAH2_STN_dehydr_react_dollar_year = \
        reactor_fixed_costs(
            react_pres_bar = FAH2_dehydr_pres_bar,
            react_vol_cu_m = FAH2_dehydr_react_vol_cu_m, 
            num_reacts = FAH2_num_dehydr_reacts, 
            output_dollar_year = output_dollar_year
            )
    
    # calculate dehydrogenation reactor installed cost ($)
    # sum of all stations
    FAH2_STN_dehydr_react_inst_cost_usd = \
        FAH2_STN_dehydr_react_inst_cost_usd_per_stn * target_num_stns

    # calculate dehydrogenation reactor O&M cost ($/yr)
    # sum of all stations
    FAH2_STN_dehydr_react_om_cost_usd_per_yr = \
        FAH2_STN_dehydr_react_om_cost_usd_per_yr_per_stn * \
        target_num_stns

    # calculate dehydrogenation reactor O&M cost ($/kg H2)
    FAH2_STN_dehydr_react_om_cost_usd_per_kg = \
        FAH2_STN_dehydr_react_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
        
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'capital cost', 
        'installed cost', 
        '$/station', 
        FAH2_STN_dehydr_react_inst_cost_usd_per_stn
        ])    
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_STN_dehydr_react_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_STN_dehydr_react_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation catalyst upfront cost

    # calculate dehydrogenation catalyst upfront purchase cost ($)
    # sum of all stations
    FAH2_STN_dehydr_catal_purc_cost_usd = \
        FAH2_dehydr_catal_cost_usd_per_kg * \
        FAH2_dehydr_catal_amt_kg * target_num_stns

    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'catalyst', 
        'capital cost', 
        'purchase cost', 
        '$/station', 
        FAH2_STN_dehydr_catal_purc_cost_usd / target_num_stns
        ])

    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station PSA *refrigerator* (precooling) energy consumption
    
    # calculate refueling station PSA *refrigerator* energy (kWh/kg H2)
    # inlet temperature = dehydrogenation reaction temperature
    # outlet temperature = PSA operating temperature
    # set refrigerator energy to zero if 
    # inlet temperature <= outlet temperature
    if FAH2_dehydr_temp_K <= FAH2_STN_psa_temp_K:
        FAH2_STN_psa_refrig_elec_kWh_per_kg = 0.0
    else:
        FAH2_STN_psa_refrig_elec_kWh_per_kg = \
            heat_exchanger_energy(
                out_temp_K = FAH2_STN_psa_temp_K,
                in_temp_K = FAH2_dehydr_temp_K
                )
    
    # convert refrigerator energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_psa_refrig_elec_MJ_per_MJ = \
        FAH2_STN_psa_refrig_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'PSA refrigerator', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        FAH2_STN_psa_refrig_elec_kWh_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'PSA refrigerator', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_STN_psa_refrig_elec_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station PSA *refrigerator* (precooling) energy emissions

    # calculate refueling station PSA *refrigerator* energy emissions 
    # (kg CO2/kg H2)
    FAH2_STN_psa_refrig_ghg_kg_CO2_per_kg = \
        FAH2_STN_psa_refrig_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert refrigerator energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_psa_refrig_ghg_g_CO2_per_MJ = \
        FAH2_STN_psa_refrig_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'PSA refrigerator', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_STN_psa_refrig_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'PSA refrigerator', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_STN_psa_refrig_ghg_g_CO2_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station PSA *refrigerator* (precooling) energy cost
    
    # calculate refueling station PSA *refrigerator* energy cost ($/kg H2)
    FAH2_STN_psa_refrig_elec_cost_usd_per_kg = \
        FAH2_STN_psa_refrig_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate refueling station refrigerator energy cost ($/yr)
    FAH2_STN_psa_refrig_elec_cost_usd_per_yr = \
        FAH2_STN_psa_refrig_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'PSA refrigerator', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        FAH2_STN_psa_refrig_elec_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'PSA refrigerator', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        FAH2_STN_psa_refrig_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station PSA *refrigerator* (precooling) installed cost and 
    # annual O&M cost

    # calculate number of PSA refrigerators needed at refueling station 
    # (= number of hoses)
    # assume linear relative to station capacity 
    # (HDSAM V3.1: 1000 kg H2/day --> 4 hoses, 4 refrigerators)
    # set number of refrigerators to zero if
    # inlet temperature <= outlet temperature
    if FAH2_dehydr_temp_K <= FAH2_STN_psa_temp_K:
        FAH2_STN_num_psa_refrigs = 0
    else: 
        FAH2_STN_num_psa_refrigs = \
            target_stn_capacity_kg_per_day / (1000.0 / 4)
    
    # calculate refueling station PSA *refrigerator* installed cost ($) 
    # and annual O&M cost ($/yr) per station, both in output dollar year 
    FAH2_STN_psa_refrig_inst_cost_usd_per_stn, \
    FAH2_STN_psa_refrig_om_cost_usd_per_yr_per_stn, \
    FAH2_STN_psa_refrig_dollar_year = \
        heat_exchanger_fixed_costs(
            out_temp_K = FAH2_STN_psa_temp_K, 
            num_hx = FAH2_STN_num_psa_refrigs, 
            output_dollar_year = output_dollar_year
            )    
    
    # calculate refueling station PSA *refrigerator* installed cost ($) 
    # sum of all stations
    FAH2_STN_psa_refrig_inst_cost_usd = \
        FAH2_STN_psa_refrig_inst_cost_usd_per_stn * target_num_stns

    # calculate refueling station PSA *refrigerator* O&M cost ($/yr)
    # sum of all stations
    FAH2_STN_psa_refrig_om_cost_usd_per_yr = \
        FAH2_STN_psa_refrig_om_cost_usd_per_yr_per_stn * target_num_stns

    # calculate refueling station PSA *refrigerator* O&M cost ($/kg H2)
    FAH2_STN_psa_refrig_om_cost_usd_per_kg = \
        FAH2_STN_psa_refrig_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'PSA refrigerator', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_STN_psa_refrig_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'PSA refrigerator', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_STN_psa_refrig_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station separator (PSA) energy consumption
    
    # initialize refueling station separator (PSA) power (kW)
    FAH2_STN_psa_power_kW = 0.0
       
    # calculate refueling station separator (PSA) power (kW)
    # TODO: calculate PSA compression power as function of dehydr. pressure
    # for now, model as step function
    if FAH2_dehydr_pres_bar < FAH2_STN_psa_pres_bar:
        
        FAH2_STN_psa_power_kW = psa_power(
            in_flow_norm_cu_m_per_hr = \
                FAH2_STN_psa_in_flow_norm_cu_m_per_hr
            )
    
    # calculate refueling station separator (PSA) energy (kWh/kg H2)
    FAH2_STN_psa_elec_kWh_per_kg = \
        FAH2_STN_psa_power_kW * target_num_stns / \
        tot_H2_deliv_kg_per_hr
    
    # convert separator (PSA) energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_psa_elec_MJ_per_MJ = \
        FAH2_STN_psa_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        FAH2_STN_psa_elec_kWh_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_STN_psa_elec_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station separator (PSA) energy emissions

    # calculate refueling station separator (PSA) energy emissions 
    # (kg CO2/kg H2)
    FAH2_STN_psa_ghg_kg_CO2_per_kg = \
        FAH2_STN_psa_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert separator (PSA) energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_psa_ghg_g_CO2_per_MJ = \
        FAH2_STN_psa_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_STN_psa_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_STN_psa_ghg_g_CO2_per_MJ
        ])

    # ------------------------------------------------------------------------
    # reconditioning - formic acid: refueling station separator energy cost
    
    # calculate refueling station separator energy cost ($/kg H2)
    FAH2_STN_psa_elec_cost_usd_per_kg = \
        FAH2_STN_psa_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate refueling station separator energy cost ($/yr)
    FAH2_STN_psa_elec_cost_usd_per_yr = \
        FAH2_STN_psa_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        FAH2_STN_psa_elec_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        FAH2_STN_psa_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station separator installed cost and annual O&M cost

    # calculate refueling station separator installed cost ($) and annual O&M
    # cost ($/yr) per station, both in output dollar year 
    FAH2_STN_psa_inst_cost_usd_per_stn, \
    FAH2_STN_psa_om_cost_usd_per_yr_per_stn, \
    FAH2_STN_psa_dollar_year = \
        psa_fixed_costs(
            in_flow_norm_cu_m_per_hr = FAH2_STN_psa_in_flow_norm_cu_m_per_hr, 
            output_dollar_year = output_dollar_year
            )
        
    # calculate refueling station separator installed cost ($)
    # sum of all stations
    FAH2_STN_psa_inst_cost_usd = \
        FAH2_STN_psa_inst_cost_usd_per_stn * target_num_stns
        
    # calculate refueling station separator O&M cost ($/yr)
    # sum of all stations
    FAH2_STN_psa_om_cost_usd_per_yr = \
        FAH2_STN_psa_om_cost_usd_per_yr_per_stn * target_num_stns
    
    # calculate refueling station separator O&M cost ($/kg H2)
    FAH2_STN_psa_om_cost_usd_per_kg = \
        FAH2_STN_psa_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_STN_psa_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation',         
        'PSA', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_STN_psa_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station compressor energy consumption and size
    
    # set compressor inlet pressure to the higher of 
    # dehydrogenation reaction pressure and PSA operating pressure
    FAH2_STN_compr_in_pres_bar = max(
        FAH2_dehydr_pres_bar, FAH2_STN_psa_pres_bar
        )
    
    # calculate compressor power (kW) and size (kW/stage) 
    # inlet pressure = reactor pressure, assuming minimal 
    # pressure drop in separator (PSA)
    FAH2_STN_compr_tot_power_kW, \
    FAH2_STN_compr_power_kW_per_stg, \
    FAH2_STN_compr_num_stgs = \
        compressor_power_and_size(
            out_pres_bar = FAH2_STN_out_pres_bar, 
            in_pres_bar = FAH2_STN_compr_in_pres_bar, 
            in_temp_K = FAH2_STN_psa_temp_K, 
            gas_flow_mol_per_sec = STN_H2_flow_mol_per_sec, 
            compressibility = 1.28
            )
    
    # calculate refueling station compressor energy (kWh/kg H2)
    FAH2_STN_compr_elec_kWh_per_kg = \
        FAH2_STN_compr_tot_power_kW * target_num_stns / \
        tot_H2_deliv_kg_per_hr
    
    # convert compressor energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_compr_elec_MJ_per_MJ = \
        FAH2_STN_compr_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        FAH2_STN_compr_elec_kWh_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_STN_compr_elec_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station compressor energy emissions

    # calculate refueling station compressor energy emissions (kg CO2/kg H2)
    FAH2_STN_compr_ghg_kg_CO2_per_kg = \
        FAH2_STN_compr_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert compressor energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_compr_ghg_g_CO2_per_MJ = \
        FAH2_STN_compr_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_STN_compr_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_STN_compr_ghg_g_CO2_per_MJ
        ])
        
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: refueling station compressor energy cost
    
    # calculate refueling station compressor energy cost ($/kg H2)
    FAH2_STN_compr_elec_cost_usd_per_kg = \
        FAH2_STN_compr_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate refueling station compressor energy cost ($/yr)
    FAH2_STN_compr_elec_cost_usd_per_yr = \
        FAH2_STN_compr_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        FAH2_STN_compr_elec_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        FAH2_STN_compr_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station compressor installed cost and annual O&M cost

    # calculate refueling station compressor installed cost ($) and annual O&M 
    # cost ($/yr) per station, both in output dollar year 
    FAH2_STN_compr_inst_cost_usd_per_stn, \
    FAH2_STN_compr_om_cost_usd_per_yr_per_stn, \
    FAH2_STN_compr_dollar_year = \
        compressor_fixed_costs(
            compr_power_kW_per_stg = FAH2_STN_compr_power_kW_per_stg, 
            num_stgs = FAH2_STN_compr_num_stgs, 
            output_dollar_year = output_dollar_year
            )
        
    # calculate refueling station compressor installed cost ($)
    # sum of all stations
    FAH2_STN_compr_inst_cost_usd = \
        FAH2_STN_compr_inst_cost_usd_per_stn * target_num_stns

    # calculate refueling station compressor O&M cost ($/yr)
    # sum of all stations
    FAH2_STN_compr_om_cost_usd_per_yr = \
        FAH2_STN_compr_om_cost_usd_per_yr_per_stn * target_num_stns 

    # calculate refueling station compressor O&M cost ($/kg H2)
    FAH2_STN_compr_om_cost_usd_per_kg = \
        FAH2_STN_compr_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_STN_compr_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_STN_compr_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station refrigerator energy consumption
    
    # calculate refueling station refrigerator energy (kWh/kg H2)
    FAH2_STN_refrig_elec_kWh_per_kg = \
        heat_exchanger_energy(
            out_temp_K = STN_dispens_temp_K,
            in_temp_K = STN_H2_temp_K
            )
    
    # convert refrigerator energy to MJ/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_refrig_elec_MJ_per_MJ = \
        FAH2_STN_refrig_elec_kWh_per_kg * MJ_per_kWh / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'energy consumption', 
        'electricity consumption', 
        'kWh/kg H2', 
        FAH2_STN_refrig_elec_kWh_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'energy consumption', 
        'electricity consumption', 
        'MJ/MJ H2 (LHV)', 
        FAH2_STN_refrig_elec_MJ_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station refrigerator energy emissions

    # calculate refueling station refrigerator energy emissions (kg CO2/kg H2)
    FAH2_STN_refrig_ghg_kg_CO2_per_kg = \
        FAH2_STN_refrig_elec_kWh_per_kg * elec_ghg_kg_CO2_per_kWh
    
    # convert refrigerator energy emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_refrig_ghg_g_CO2_per_MJ = \
        FAH2_STN_refrig_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'emissions', 
        'energy emissions', 
        'kg CO2/kg H2', 
        FAH2_STN_refrig_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'emissions', 
        'energy emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_STN_refrig_ghg_g_CO2_per_MJ
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station refrigerator energy cost
    
    # calculate refueling station refrigerator energy cost ($/kg H2)
    FAH2_STN_refrig_elec_cost_usd_per_kg = \
        FAH2_STN_refrig_elec_kWh_per_kg * elec_cost_usd_per_kWh
    
    # calculate refueling station refrigerator energy cost ($/yr)
    FAH2_STN_refrig_elec_cost_usd_per_yr = \
        FAH2_STN_refrig_elec_cost_usd_per_kg * tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'energy cost', 
        'electricity cost', 
        '$/yr', 
        FAH2_STN_refrig_elec_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'energy cost', 
        'electricity cost', 
        '$/kg H2', 
        FAH2_STN_refrig_elec_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station refrigerator installed cost and annual O&M cost
        
    # calculate number of refrigerators needed at refueling station 
    # (= number of hoses)
    # assume linear relative to station capacity 
    # (HDSAM V3.1: 1000 kg H2/day --> 4 hoses, 4 refrigerators)
    FAH2_STN_num_refrigs = \
        target_stn_capacity_kg_per_day / (1000.0 / 4)
    
    # calculate refueling station refrigerator installed cost ($) and annual 
    # O&M cost ($/yr) per station, both in output dollar year 
    FAH2_STN_refrig_inst_cost_usd_per_stn, \
    FAH2_STN_refrig_om_cost_usd_per_yr_per_stn, \
    FAH2_STN_refrig_dollar_year = \
        heat_exchanger_fixed_costs(
            out_temp_K = STN_dispens_temp_K, 
            num_hx = FAH2_STN_num_refrigs, 
            output_dollar_year = output_dollar_year
            )    
        
    # calculate refueling station refrigerator installed cost ($) 
    # sum of all stations
    FAH2_STN_refrig_inst_cost_usd = \
        FAH2_STN_refrig_inst_cost_usd_per_stn * target_num_stns

    # calculate refueling station refrigerator O&M cost ($/yr)
    # sum of all stations
    FAH2_STN_refrig_om_cost_usd_per_yr = \
        FAH2_STN_refrig_om_cost_usd_per_yr_per_stn * target_num_stns 

    # calculate refueling station refrigerator O&M cost ($/kg H2)
    FAH2_STN_refrig_om_cost_usd_per_kg = \
        FAH2_STN_refrig_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_STN_refrig_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_STN_refrig_om_cost_usd_per_kg
        ])
    

    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station formic acid storage installed cost and annual O&M cost
    
    # calculate total formic acid storage capacity (m^3) and 
    # number of storage tanks required at refueling station
    FAH2_STN_stor_tank_capacity_cu_m, \
    FAH2_STN_num_tanks = \
        general_tank_storage_size(
            fluid_flow_kg_per_day = FAH2_STN_FA_flow_kg_per_day,
            stor_amt_days = FAH2_STN_stor_amt_days,
            fluid_dens_kg_per_cu_m = dens_FA_kg_per_cu_m
            )
    
    # calculate formic acid storage installed cost ($) and annual O&M 
    # cost ($/yr) per station, both in output dollar year
    FAH2_STN_stor_inst_cost_usd_per_stn, \
    FAH2_STN_stor_om_cost_usd_per_yr_per_stn, \
    FAH2_STN_stor_dollar_year = \
        general_tank_stor_fixed_costs(
            stor_tank_capacity_cu_m = FAH2_STN_stor_tank_capacity_cu_m,
            num_tanks = FAH2_STN_num_tanks,
            output_dollar_year = output_dollar_year, 
            material = 'fiber glass open top'
            )
        
    # calculate formic acid storage installed cost ($)
    # sum of all stations
    FAH2_STN_stor_inst_cost_usd = \
        FAH2_STN_stor_inst_cost_usd_per_stn * target_num_stns

    # calculate formic acid storage O&M cost ($/yr)
    # sum of all stations
    FAH2_STN_stor_om_cost_usd_per_yr = \
        FAH2_STN_stor_om_cost_usd_per_yr_per_stn * target_num_stns
                
    # calculate formic acid storage O&M cost ($/kg H2)
    FAH2_STN_stor_om_cost_usd_per_kg = \
        FAH2_STN_stor_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_STN_stor_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_STN_stor_om_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station cascade storage installed cost and annual O&M cost

    # calculate total cascade storage capacity required (kg) at 
    # refueling station
    FAH2_STN_casc_stor_tot_capacity_kg = \
        station_cascade_storage_size(
            stn_capacity_kg_per_day = target_stn_capacity_kg_per_day,
            casc_stor_size_frac = FAH2_STN_casc_stor_size_frac
            )
    
    # calculate cascade storage installed cost ($) and annual O&M 
    # cost ($/yr) per station, both in output dollar year
    FAH2_STN_casc_stor_inst_cost_usd_per_stn, \
    FAH2_STN_casc_stor_om_cost_usd_per_yr_per_stn, \
    FAH2_STN_casc_stor_dollar_year = \
        station_cascade_storage_fixed_costs(
            stor_tot_capacity_kg = FAH2_STN_casc_stor_tot_capacity_kg, 
            output_dollar_year = output_dollar_year
            )
        
    # calculate cascade storage installed cost ($)
    # sum of all stations
    FAH2_STN_casc_stor_inst_cost_usd = \
        FAH2_STN_casc_stor_inst_cost_usd_per_stn * target_num_stns

    # calculate cascade storage O&M cost ($/yr)
    # sum of all stations
    FAH2_STN_casc_stor_om_cost_usd_per_yr = \
        FAH2_STN_casc_stor_om_cost_usd_per_yr_per_stn * target_num_stns
                
    # calculate cascade storage O&M cost ($/kg H2)
    FAH2_STN_casc_stor_om_cost_usd_per_kg = \
        FAH2_STN_casc_stor_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/yr', 
        FAH2_STN_casc_stor_om_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'operation, maintenance, repair costs', 
        '$/kg H2', 
        FAH2_STN_casc_stor_om_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station total capital investment and "other" annual O&M costs
        
    # "Other" O&M costs include insurance, property taxes, licensing and 
    # permits. Operation, maintenance, and repair costs for individual 
    # refueling station components (e.g., compressor) are calculated 
    # separately.
        
    # calculate refueling station total initial capital investment ($)
    # (= dehydrogenation pump + reactor + catalyst + 
    # separator *refrigerator* + separator + compressor + refrigerator + 
    # formic acid storage + cascade storage 
    # for *formic acid* hydrogen refueling station)
    # TODO: revisit components at LOHC / formic acid refueling station
    # sum of all stations
    FAH2_STN_init_cap_inv_usd = \
        FAH2_STN_dehydr_pump_inst_cost_usd + \
        FAH2_STN_dehydr_react_inst_cost_usd + \
        FAH2_STN_dehydr_catal_purc_cost_usd + \
        FAH2_STN_psa_refrig_inst_cost_usd + \
        FAH2_STN_psa_inst_cost_usd + \
        FAH2_STN_compr_inst_cost_usd + \
        FAH2_STN_refrig_inst_cost_usd + \
        FAH2_STN_stor_inst_cost_usd + \
        FAH2_STN_casc_stor_inst_cost_usd
    
    # calculate refueling station cost allocations (%) to dehydrogenation pump, 
    # reactor, catalyst, separator *refrigerator*, separator, compressor, 
    # refrigerator, and cascade storage
    # % of refueling station total initial capital investment
    # use to allocate total capital investment, other O&M costs, and labor 
    # cost
    FAH2_STN_dehydr_pump_cost_perc = \
        FAH2_STN_dehydr_pump_inst_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    FAH2_STN_dehydr_react_cost_perc = \
        FAH2_STN_dehydr_react_inst_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    FAH2_STN_dehydr_catal_cost_perc = \
        FAH2_STN_dehydr_catal_purc_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    FAH2_STN_psa_refrig_cost_perc = \
        FAH2_STN_psa_refrig_inst_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    FAH2_STN_psa_cost_perc = \
        FAH2_STN_psa_inst_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    FAH2_STN_compr_cost_perc = \
        FAH2_STN_compr_inst_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    FAH2_STN_refrig_cost_perc = \
        FAH2_STN_refrig_inst_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    FAH2_STN_stor_cost_perc = \
        FAH2_STN_stor_inst_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    FAH2_STN_casc_stor_cost_perc = \
        FAH2_STN_casc_stor_inst_cost_usd / \
        FAH2_STN_init_cap_inv_usd
    
    # check whether cost allocations (%) sum to one
    # raise error if false
    if abs(
            FAH2_STN_dehydr_pump_cost_perc + \
            FAH2_STN_dehydr_react_cost_perc + \
            FAH2_STN_dehydr_catal_cost_perc + \
            FAH2_STN_psa_refrig_cost_perc + \
            FAH2_STN_psa_cost_perc + \
            FAH2_STN_compr_cost_perc + \
            FAH2_STN_refrig_cost_perc + \
            FAH2_STN_stor_cost_perc + \
            FAH2_STN_casc_stor_cost_perc - \
            1.0
            ) >= 1.0e-9:
        raise ValueError(
            'Component cost allocations need to sum to one.'
            )
        
    # check if all refueling station components have the same dollar year
    # if true, assign dollar year of refueling station costs to the dollar 
    # year of one of the components 
    if (FAH2_STN_dehydr_pump_dollar_year == \
        FAH2_STN_dehydr_react_dollar_year) \
        and (FAH2_STN_dehydr_react_dollar_year == \
             FAH2_STN_psa_refrig_dollar_year) \
        and (FAH2_STN_psa_refrig_dollar_year == \
             FAH2_STN_psa_dollar_year) \
        and (FAH2_STN_psa_dollar_year == \
             FAH2_STN_compr_dollar_year) \
        and (FAH2_STN_compr_dollar_year == \
             FAH2_STN_refrig_dollar_year) \
        and (FAH2_STN_refrig_dollar_year == \
             FAH2_STN_stor_dollar_year) \
        and (FAH2_STN_stor_dollar_year == \
             FAH2_STN_casc_stor_dollar_year):
        FAH2_STN_dollar_year = FAH2_STN_dehydr_react_dollar_year    
    else:
        raise ValueError(
            'Dollar year of components need to match.'
            )
        
    # calculate refueling station total capital investment 
    # ($, output dollar year), sum of all stations
    FAH2_STN_tot_cap_inv_usd, \
    FAH2_STN_cap_cost_dollar_year = \
        station_total_capital_investment(
            init_cap_inv_usd = FAH2_STN_init_cap_inv_usd, 
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate refueling station "other" annual O&M costs 
    # ($/yr, output dollar year), sum of all stations
    FAH2_STN_om_cost_usd_per_yr, \
    FAH2_STN_om_cost_dollar_year = \
        other_om_cost(
            tot_cap_inv_usd = FAH2_STN_tot_cap_inv_usd,
            input_dollar_year = FAH2_STN_cap_cost_dollar_year
            )
    
    # calculate refueling station "other" O&M costs ($/kg H2)
    FAH2_STN_om_cost_usd_per_kg = \
        FAH2_STN_om_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign "other" O&M costs to dehydrogenation pump, reactor, catalyst,
    # separator *refrigerator*, seprarator, compressor, 
    # refrigerator, formic acid storage, and cascade storage
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_dehydr_pump_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_dehydr_pump_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_dehydr_react_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_dehydr_react_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'catalyst', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_dehydr_catal_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'catalyst', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_dehydr_catal_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA refrigerator', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_psa_refrig_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA refrigerator', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_psa_refrig_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_psa_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_psa_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_compr_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_compr_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_refrig_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_refrig_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_stor_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_stor_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/yr', 
        FAH2_STN_om_cost_usd_per_yr * \
            FAH2_STN_casc_stor_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'other O&M costs', 
        '$/kg H2', 
        FAH2_STN_om_cost_usd_per_kg * \
            FAH2_STN_casc_stor_cost_perc
        ])
            
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: refueling station annual labor cost
    
    # NOTE: HDSAM V3.1 refueling station labor costs are derived from gasoline 
    # stations and do not have the need for onsite dehydrogenation and
    # separation. Using HDSAM V3.1 formula likely underestimates the labor 
    # cost for LOHC / formic acid refueling stations.
    
    # TODO: revisit labor cost scaling for formic acid refueling station
    
    # calculate refueling station annual labor cost ($/yr, output dollar 
    # year) per station, including overhead and G&A    
    FAH2_STN_labor_cost_usd_per_yr_per_stn, \
    FAH2_STN_labor_cost_dollar_year = \
        station_labor_cost(
            H2_flow_kg_per_day = target_stn_capacity_kg_per_day, 
            output_dollar_year = output_dollar_year
            ) 
    
    # calculate refueling station labor cost ($/yr)
    # sum of all stations
    FAH2_STN_labor_cost_usd_per_yr = \
        FAH2_STN_labor_cost_usd_per_yr_per_stn * target_num_stns
    
    # calculate refueling station labor cost ($/kg H2)
    FAH2_STN_labor_cost_usd_per_kg = \
        FAH2_STN_labor_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    # assign labor cost to dehydrogenation pump, reactor, catalyst,
    # separator *refrigerator*, seprarator, compressor, refrigerator, 
    # formic acid storage, and cascade storage
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_dehydr_pump_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_dehydr_pump_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_dehydr_react_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_dehydr_react_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'catalyst', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_dehydr_catal_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'catalyst', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_dehydr_catal_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA refrigerator', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_psa_refrig_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA refrigerator', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_psa_refrig_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_psa_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_psa_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_compr_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_compr_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_refrig_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_refrig_cost_perc
        ])

    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_stor_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'liquid storage', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_stor_cost_perc
        ])
        
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'labor cost', 
        '$/yr', 
        FAH2_STN_labor_cost_usd_per_yr * \
            FAH2_STN_casc_stor_cost_perc
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'O&M cost', 
        'labor cost', 
        '$/kg H2', 
        FAH2_STN_labor_cost_usd_per_kg * \
            FAH2_STN_casc_stor_cost_perc
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation pump levelized capital cost
    
    # calculate dehydrogenation pump total capital investment ($) 
    # (= refueling station total capital investment allocated to 
    # dehydrogenation pump)
    # sum of all stations
    FAH2_STN_dehydr_pump_tot_cap_inv_usd = \
        FAH2_STN_dehydr_pump_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate dehydrogenation pump levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_dehydr_pump_lev_cap_cost_usd_per_yr, \
    FAH2_STN_dehydr_pump_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_dehydr_pump_tot_cap_inv_usd, 
            life_yr = STN_pump_life_yr, 
            depr_yr = STN_pump_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate dehydrogenation pump levelized capital cost ($/kg H2)
    FAH2_STN_dehydr_pump_lev_cap_cost_usd_per_kg = \
        FAH2_STN_dehydr_pump_lev_cap_cost_usd_per_yr / \
        tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_dehydr_pump_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'pumping', 
        'reactor pump', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_dehydr_pump_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation reactor levelized capital cost
    
    # calculate reactor total capital investment ($) 
    # (= refueling station total capital investment allocated to 
    # dehydrogenation reactor)
    # sum of all stations
    FAH2_STN_dehydr_react_tot_cap_inv_usd = \
        FAH2_STN_dehydr_react_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate reactor levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_dehydr_react_lev_cap_cost_usd_per_yr, \
    FAH2_STN_dehydr_react_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_dehydr_react_tot_cap_inv_usd, 
            life_yr = STN_react_life_yr, 
            depr_yr = STN_react_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate reactor levelized capital cost ($/kg H2)
    FAH2_STN_dehydr_react_lev_cap_cost_usd_per_kg = \
        FAH2_STN_dehydr_react_lev_cap_cost_usd_per_yr / \
        tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_dehydr_react_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_dehydr_react_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # dehydrogenation catalyst levelized capital cost

    # calculate dehydrogenation catalyst total capital 
    # investment ($) 
    # (= refueling station total capital investment allocated to 
    # dehydrogenation catalyst)
    # sum of all stations
    FAH2_STN_dehydr_catal_tot_cap_inv_usd = \
        FAH2_STN_dehydr_catal_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate dehydrogenation catalyst levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_dehydr_catal_lev_cap_cost_usd_per_yr, \
    FAH2_STN_dehydr_catal_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_dehydr_catal_tot_cap_inv_usd, 
            life_yr = FAH2_dehydr_catal_life_yr, 
            depr_yr = FAH2_dehydr_catal_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate dehydrogenation catalyst levelized capital 
    # cost ($/kg H2)
    FAH2_STN_dehydr_catal_lev_cap_cost_usd_per_kg = \
        FAH2_STN_dehydr_catal_lev_cap_cost_usd_per_yr / \
        tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'catalyst', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_dehydr_catal_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'catalyst', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_dehydr_catal_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station PSA *refrigerator* (precooling) levelized capital cost

    # calculate refueling station PSA *refrigerator* total 
    # capital investment ($) 
    # (= refueling station total capital investment allocated to 
    # PSA *refrigerator*)
    # sum of all stations
    FAH2_STN_psa_refrig_tot_cap_inv_usd = \
        FAH2_STN_psa_refrig_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate refueling station PSA *refrigerator* levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_psa_refrig_lev_cap_cost_usd_per_yr, \
    FAH2_STN_psa_refrig_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_psa_refrig_tot_cap_inv_usd, 
            life_yr = STN_refrig_life_yr, 
            depr_yr = STN_refrig_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate refueling station PSA *refrigerator* levelized capital cost 
    # ($/kg H2)
    FAH2_STN_psa_refrig_lev_cap_cost_usd_per_kg = \
        FAH2_STN_psa_refrig_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA refrigerator', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_psa_refrig_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA refrigerator', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_psa_refrig_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station separator levelized capital cost
    
    # calculate refueling station separator (PSA) total capital investment ($) 
    # (= refueling station total capital investment allocated to separator)
    # sum of all stations
    FAH2_STN_psa_tot_cap_inv_usd = \
        FAH2_STN_psa_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate refueling station separator (PSA) levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_psa_lev_cap_cost_usd_per_yr, \
    FAH2_STN_psa_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_psa_tot_cap_inv_usd, 
            life_yr = STN_psa_life_yr, 
            depr_yr = STN_psa_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate refueling station separator (PSA) levelized capital cost 
    # ($/kg H2)
    FAH2_STN_psa_lev_cap_cost_usd_per_kg = \
        FAH2_STN_psa_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_psa_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'separation', 
        'PSA', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_psa_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station compressor levelized capital cost
    
    # calculate refueling station compressor total capital investment ($) 
    # (= refueling station total capital investment allocated to compressor)
    # sum of all stations
    FAH2_STN_compr_tot_cap_inv_usd = \
        FAH2_STN_compr_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate refueling station compressor levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_compr_lev_cap_cost_usd_per_yr, \
    FAH2_STN_compr_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_compr_tot_cap_inv_usd, 
            life_yr = STN_compr_life_yr, 
            depr_yr = STN_compr_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate refueling station compressor levelized capital cost ($/kg H2)
    FAH2_STN_compr_lev_cap_cost_usd_per_kg = \
        FAH2_STN_compr_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_compr_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'compression', 
        'compressor', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_compr_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station refrigerator levelized capital cost

    # calculate refueling station refrigerator total capital investment ($) 
    # (= refueling station total capital investment allocated to refrigerator)
    # sum of all stations
    FAH2_STN_refrig_tot_cap_inv_usd = \
        FAH2_STN_refrig_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate refueling station refrigerator levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_refrig_lev_cap_cost_usd_per_yr, \
    FAH2_STN_refrig_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_refrig_tot_cap_inv_usd, 
            life_yr = STN_refrig_life_yr, 
            depr_yr = STN_refrig_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate refueling station refrigerator levelized capital cost 
    # ($/kg H2)
    FAH2_STN_refrig_lev_cap_cost_usd_per_kg = \
        FAH2_STN_refrig_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_refrig_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'cooling', 
        'refrigerator', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_refrig_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station formic acid storage levelized capital cost
    
    # calculate refueling station formic acid storage total 
    # capital investment ($) 
    # (= refueling station total capital investment allocated to formic acid
    # storage)
    # sum of all stations
    FAH2_STN_stor_tot_cap_inv_usd = \
        FAH2_STN_stor_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate refueling station formic acid storage levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_stor_lev_cap_cost_usd_per_yr, \
    FAH2_STN_stor_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_stor_tot_cap_inv_usd, 
            life_yr = STN_stor_life_yr, 
            depr_yr = STN_stor_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate refueling station formic acid storage levelized capital cost 
    # ($/kg H2)
    FAH2_STN_stor_lev_cap_cost_usd_per_kg = \
        FAH2_STN_stor_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'liquid storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_stor_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'liquid storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_stor_lev_cap_cost_usd_per_kg
        ])

    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station cascade storage levelized capital cost
    
    # calculate refueling station cascade storage total capital investment ($) 
    # (= refueling station total capital investment allocated to cascade
    # storage)
    # sum of all stations
    FAH2_STN_casc_stor_tot_cap_inv_usd = \
        FAH2_STN_casc_stor_cost_perc * FAH2_STN_tot_cap_inv_usd
    
    # calculate refueling station cascade storage levelized capital cost 
    # ($/yr, output dollar year)
    # sum of all stations
    FAH2_STN_casc_stor_lev_cap_cost_usd_per_yr, \
    FAH2_STN_casc_stor_lev_cap_cost_dollar_year = \
        levelized_capital_cost(
            tot_cap_inv_usd = FAH2_STN_casc_stor_tot_cap_inv_usd, 
            life_yr = STN_stor_life_yr, 
            depr_yr = STN_stor_depr_yr,
            input_dollar_year = FAH2_STN_dollar_year
            )
    
    # calculate refueling station cascade storage levelized capital cost 
    # ($/kg H2)
    FAH2_STN_casc_stor_lev_cap_cost_usd_per_kg = \
        FAH2_STN_casc_stor_lev_cap_cost_usd_per_yr / tot_H2_deliv_kg_per_yr
    
    # append results to list
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/yr', 
        FAH2_STN_casc_stor_lev_cap_cost_usd_per_yr
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'storage', 
        'cascade storage', 
        'capital cost', 
        'levelized capital cost', 
        '$/kg H2', 
        FAH2_STN_casc_stor_lev_cap_cost_usd_per_kg
        ])
    
    # ------------------------------------------------------------------------
    # reconditioning - formic acid: 
    # refueling station process emissions 
    # (separator outlet, if CO2 is not captured)

    # TODO: revisit process emissions with updated reaction yield
        
    # initialize dehydrogenation process emissions (zero by default)
    FAH2_STN_proc_ghg_kg_CO2_per_kg = 0.0

    # if purchase formic acid, calculate dehydrogenation process emissions
    if FA_prod_pathway == 'purchase':
        
        # calculate dehydrogenation process emissions (kg CO2/kg H2)
        FAH2_STN_proc_ghg_kg_CO2_per_kg = \
            stoic_mol_CO2_per_mol_FA * molar_mass_CO2_kg_per_kmol / (
            stoic_mol_H2_per_mol_FA * molar_mass_H2_kg_per_kmol
            )
        
    # convert dehydrogenation process emissions to g CO2/MJ H2 (LHV) 
    # (for comparison with HDSAM V3.1)
    FAH2_STN_proc_ghg_g_CO2_per_MJ = \
        FAH2_STN_proc_ghg_kg_CO2_per_kg * g_per_kg / \
        low_heat_val_H2_MJ_per_kg
    
    # append results to list
    # for now, attribute emissions to reaction / reactor
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'emissions', 
        'process emissions', 
        'kg CO2/kg H2', 
        FAH2_STN_proc_ghg_kg_CO2_per_kg
        ])
    list_output.append([
        'LOHC - formic acid', 
        'reconditioning', 
        'refueling station', 
        'reaction', 
        'reactor', 
        'emissions', 
        'process emissions', 
        'g CO2/MJ H2 (LHV)', 
        FAH2_STN_proc_ghg_g_CO2_per_MJ
        ])

    #%% SAVE RESULTS
        
    # convert inputs and results list to dataframe
    df_output = pd.DataFrame(list_output, columns = output_columns)
    
    # define custom categories for sorting results
    pathway_categories = [
        'all', 
        'compressed hydrogen', 
        'liquid hydrogen', 
        'LOHC - formic acid'
        ]
    process_categories = [
        'all', 
        'production',
        'preconditioning', 
        'transport', 
        'reconditioning'
        ]
    variable_group_categories = [
        'input parameter', 
        'energy consumption', 
        'emissions', 
        'capital cost', 
        'O&M cost', 
        'energy cost'
        ]
    
    # sort results by custom categories
    df_output_sorted = df_output.copy()
    df_output_sorted['pathway'] = pd.Categorical(
        df_output_sorted['pathway'], 
        categories = pathway_categories
        )
    df_output_sorted['process'] = pd.Categorical(
        df_output_sorted['process'], 
        categories = process_categories
        )
    df_output_sorted['variable group'] = pd.Categorical(
        df_output_sorted['variable group'], 
        categories = variable_group_categories
        )
    df_output_sorted = \
        df_output_sorted.sort_values(by = [
            'pathway', 
            'process', 
            'variable group'
            ])
        
    # filter for $/kg H2 cost results
    df_costs_usd_per_kg = df_output.loc[
        df_output['unit'] == '$/kg H2'
        ].reset_index(drop = True)
    
    # calculate total $/kg H2 cost by pathway
    df_tot_costs_usd_per_kg = \
        df_costs_usd_per_kg.groupby(
            by = ['pathway'])['value'].sum().reset_index()
        
    # assign total $/kg H2 costs to each pathway
    FAH2_tot_H2_cost_usd_per_kg = df_tot_costs_usd_per_kg.loc[
            df_tot_costs_usd_per_kg['pathway'] == 'LOHC - formic acid', 
            'value'
            ].values[0]
    GH2_tot_H2_cost_usd_per_kg = df_tot_costs_usd_per_kg.loc[
            df_tot_costs_usd_per_kg['pathway'] == 'compressed hydrogen', 
            'value'
            ].values[0]
    LH2_tot_H2_cost_usd_per_kg = df_tot_costs_usd_per_kg.loc[
            df_tot_costs_usd_per_kg['pathway'] == 'liquid hydrogen', 
            'value'
            ].values[0]
                    
    if save_csv == True:
        
        # create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # save inputs and results dataframe
        df_output_sorted.to_csv(
            os.path.join(
                output_folder, 
                'output_' + str(run_num).zfill(4) + '.csv'
                ),         
        index = False)
            
    return df_output, \
        FAH2_tot_H2_cost_usd_per_kg, \
        GH2_tot_H2_cost_usd_per_kg, \
        LH2_tot_H2_cost_usd_per_kg
        
#%% RUN TEST

if __name__ == '__main__':
    df = calcs()
