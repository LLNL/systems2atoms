# Documentation

This document provides an overview of the functions available in `s2a_systems_functions.py`.

## Function: mol_to_norm_cu_m

Calculate gas volume at normal conditions (Nm^3) for given number of moles. Works for molar flowrate to volumetric flowrate conversion if time unit is the same (e.g., mol/hr --> Nm^3/hr).

Parameters
----------
num_mols : float
    Number of moles of gas

Returns
-------
gas_vol_norm_cu_m
    Gas volume (Nm^3) (or volumetric flowrate, Nm^3/time)

---

## Function: dollar_year_conversion

Calculate dollar year conversion (multiplier) using user-selected cost index. Either input dollar year (e.g., 2006) or input cost index (e.g., CEPCI = 1000) needs to be provided.

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

---

## Function: MACRS_depreciation_NPV

Calculate net present value (NPV) in % using MACRS depreciation.

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

---

## Function: levelized_capital_cost

Calculate levelized capital cost ($/yr, specified output dollar year).

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

---

## Function: compressor_power_and_size

Calculate compressor power (kW), size (kW/pump), number of compressor stages. Outputs validated using HDSAM V3.1.

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

---

## Function: compressor_fixed_costs

Calculate compressor installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Compressor installed cost (\\$, user-specified output dollar year). 
compr_om_cost_usd_per_yr
    Compressor annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: pump_power_and_size

Calculate pump power (kW), size (kW/pump), number of pumps. Outputs validated using HDSAM V3.1.

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

---

## Function: cryo_pump_fixed_costs

Calculate high-pressure cryogenic pump installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Pump installed cost (\\$, user-specified output dollar year). 
pump_om_cost_usd_per_yr
    Pump annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: low_head_pump_fixed_costs

Calculate low-head pump installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year. Outputs validated using HDSAM V3.1.

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
    Pump installed cost (\\$, user-specified output dollar year). 
pump_om_cost_usd_per_yr
    Pump annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: vaporizer_fixed_costs

Calculate vaporizer (evaporator) installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Vaporizer installed cost (\\$, user-specified output dollar year). 
vap_om_cost_usd_per_yr
    Vaporizer annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: heat_exchanger_energy

Calculate heat exchanger energy (kWh/kg fluid). Outputs validated using HDSAM V3.1.

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

---

## Function: heat_exchanger_fixed_costs

Calculate heat exchanger installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Heat exchanger installed cost (\\$, user-specified output dollar year). 
hx_om_cost_usd_per_yr
    Heat exchanger annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: liquefier_energy

Calculate liquefier energy (kWh/kg H2). Varies with assumed liquefier size (tonne H2/day), using relationship in HDSAM V3.1. Outputs validated using HDSAM V3.1.

Parameters
----------
liquef_size_tonne_per_day : float
    Liquefier size (capacity) (tonne H2/day).

Returns
-------
liquef_elec_kWh_per_kg
    Liquefier energy (kWh/kg H2)

---

## Function: liquefier_fixed_costs

Calculate liquefier installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year. Installed cost scales with assumed liquefier size (tonne H2/day), using relationship in HDSAM V3.1. Outputs validated using HDSAM V3.1.
        
Parameters
----------
liquef_size_tonne_per_day : float
    Liquefier size (capacity) (tonne H2/day).
output_dollar_year : int
    Dollar year of calculated costs.    
    
Returns
-------
liquef_inst_cost_usd
    Liquefier installed cost (\\$, user-specified output dollar year). 
liquef_om_cost_usd_per_yr
    Liquefier annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: reactor_fixed_costs

Calculate reactor installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Reactor installed cost (\\$, user-specified output dollar year). 
react_om_cost_usd_per_yr
    Reactor annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: electrolyzer_power

Calculate electrolyzer power (kW) for a target output flowrate.

Parameters
----------
electr_volt_V : float
    Applied electrolyzer cell voltage (V).
electr_curr_dens_A_per_sq_m : float
    Electrolyzer current density (A/m^2).
electr_area_sq_m_per_cell : float
    Electrolyzer area per cell (m^2/cell).         
out_flow_kg_per_sec_per_cell : float
    Electrolyzer output flowrate per cell (kg/s-cell).
target_out_flow_kg_per_sec : float
    Total target output flowrate (kg/s).
    
Returns
-------
electr_power_kW
    Electrolyzer total power (kW) for the target output flowrate.

---

## Function: electrolyzer_fixed_costs

Calculate electrolyzer installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.
        
Reference(s): 
(a) Ramdin, M., Morrison, A. R. T., de Groen, M., van Haperen, R., de Kler, R., van den Broeke, L. J. P., Trusler, J. P. M., de Jong, W., & Vlugt, T. J. H. (2019). High Pressure Electrochemical Reduction of CO2 to Formic Acid/Formate: A Comparison between Bipolar Membranes and Cation Exchange Membranes. Industrial & Engineering Chemistry Research, 58(5), 1834-1847. https://doi.org/10.1021/acs.iecr.8b04944.
(b) Li, W. Q., J. T. Feaster, S. A. Akhade, J. T. Davis, A. A. Wong, V. A. Beck, . . . S. E. Baker. 2021. "Comparative Techno-Economic and Life Cycle Analysis of Water Oxidation and Hydrogen Oxidation at the Anode in a CO2 Electrolysis to Ethylene System." ACS Sustainable Chemistry & Engineering 9 (44): 14678-14689. https://doi.org/10.1021/acssuschemeng.1c01846.


Parameters
----------
electr_area_sq_m_per_cell : float
    Electrolyzer area per cell (m^2/cell). 
out_flow_kg_per_sec_per_cell : float
    Electrolyzer output flowrate per cell (kg/s-cell).
target_out_flow_kg_per_sec : float
    Total target output flowrate (kg/s).
output_dollar_year : int
    Dollar year of calculated costs. 
electr_purc_cost_usd_per_sq_m : float, optional
    Electrolyzer purchase cost ($/m^2). Default: CO2 electrolyzer in Ramdin et al., 2019.
electr_cost_dollar_year : int, optional
    Input dollar year of electrolyzer cost. Default: 2019 (publication year of Ramdin et al., 2019).
    
Returns
-------
electr_inst_cost_usd
    Electrolyzer installed cost (\\$, user-specified output dollar year). 
electr_om_cost_usd_per_yr
    Electrolyzer annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: psa_power

Calculate separator (pressure swing adsorption, PSA) power (kW).

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

---

## Function: psa_fixed_costs

Calculate separator (pressure swing adsorption, PSA) installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Separator (PSA) installed cost (\\$, user-specified output dollar year). 
psa_om_cost_usd_per_yr
    Separator (PSA) annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: transport_energy

Calculate transport fuel consumption (gallon/kg cargo), number of vehicles required, number of deliveries per day, total trip time (hr/trip). Outputs validated using HDSAM V3.1.

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

---

## Function: gas_truck_capital_cost

Calculate compressed gaseous hydrogen truck capital cost ($) in user-specified output dollar year.

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

---

## Function: liquid_truck_capital_cost

Calculate liquid hydrogen truck capital cost ($) in user-specified output dollar year.

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

---

## Function: truck_om_cost

Calculate truck annual O&M cost ($/yr) in user-specified output dollar year. Outputs validated using HDSAM V3.1.

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

---

## Function: truck_labor_cost

Calculate truck annual labor cost ($/yr) in user-specified output dollar year. Outputs validated using HDSAM V3.1.

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

---

## Function: CO2_transport_all_in_cost

Interpolate all-in liquid CO2 trucking cost ($) for given amount of CO2 (kt/yr) and transport distance (mile). Output is in user-specified output dollar year.

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
    All-in liquid CO2 trucking cost (\\$/tonne CO2, user-specified output dollar year). 
liq_CO2_trucking_cost_usd_per_yr
    All-in liquid CO2 trucking cost (\$/yr, user-specified output dollar year). 
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: GH2_terminal_storage_size

Calculate compressed hydrogen terminal storage total capacity (kg) and number of storage tanks required. Outputs validated using HDSAM V3.1.

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

---

## Function: LH2_terminal_storage_size

Calculate liquid hydrogen terminal storage design capacity per tank (m^3) and number of storage tanks required. Outputs validated using HDSAM V3.1.

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

---

## Function: LH2_station_cryo_storage_size

Calculate liquid hydrogen refueling station cryogenic storage capacity (kg), with capacity subject to available (discrete) tank capacities. Outputs validated using HDSAM V3.1.

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

---

## Function: station_cascade_storage_size

Calculate refueling station cascade storage capacity (kg).

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

---

## Function: GH2_terminal_storage_fixed_costs

Calculate compressed hydrogen terminal storage installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Compressed hydrogen terminal storage installed cost (\\$, user-specified output dollar year). 
stor_om_cost_usd_per_yr
    Compressed hydrogen terminal storage annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: LH2_terminal_storage_fixed_costs

Calculate liquid hydrogen terminal storage installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Liquid hydrogen terminal storage installed cost (\\$, user-specified output dollar year). 
stor_om_cost_usd_per_yr
    Liquid hydrogen terminal storage annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: LH2_station_cryo_storage_fixed_costs

Calculate liquid hydrogen refueling station cryogenic storage installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.
        
Parameters
----------
stor_tot_capacity_kg : float
    Total capacity of cryogenic storage tank (kg) required at liquid hydrogen refueling station.
output_dollar_year : int
    Dollar year of calculated costs.

Returns
-------
stor_inst_cost_usd
    Liquid hydrogen refueling station cryogenic storage installed cost (\\$, user-specified output dollar year). 
stor_om_cost_usd_per_yr
    Liquid hydrogen refueling station cryogenic storage annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: station_cascade_storage_fixed_costs

Calculate refueling station cascade storage installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.

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
    Refueling station cascade storage installed cost (\\$, user-specified output dollar year). 
stor_om_cost_usd_per_yr
    Refueling station cascade storage annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: general_tank_storage_size

Calculate general-purpose storage tank design capacity per tank (m^3) and number of storage tanks required.

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

---

## Function: general_tank_stor_fixed_costs

Calculate general-purpose storage tank installed cost (\\$) and annual O&M cost (\$/yr), both in user-specified output dollar year.
        
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
    Storage tank installed cost (\\$, user-specified output dollar year). 
stor_tank_om_cost_usd_per_yr
    Storage tank annual O&M cost (\$/yr, user-specified output dollar year).
output_dollar_year
    User-specified output dollar year, for sanity check.

---

## Function: non_station_total_capital_investment

Calculate non-refueling station (e.g., terminal) total capital investment ($) in user-specified output dollar year.

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

---

## Function: station_total_capital_investment

Calculate refueling station total capital investment ($) in user-specified output dollar year.

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

---

## Function: other_om_cost

Calculate other O&M costs ($/yr) for refueling station or non-refueling station (e.g., terminal, liquefier) in user-specified output dollar year.

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

---

## Function: non_station_labor_cost

Calculate non-refueling station annual labor cost ($/yr) in user-specified output dollar year. Outputs validated using HDSAM V3.1.

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

---

## Function: station_labor_cost

Calculate refueling station annual labor cost ($/yr) in user-specified output dollar year. Outputs validated using HDSAM V3.1.

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

---

## Function: calcs

Run S2A systems technoeconomic analysis for various delivery pathways.

Parameters
----------
dict_input_params : dict
    Dictionary containing one set of input parameters for one run.
save_csv : bool, default False
    If True, save output csv file. Otherwise return total hydrogen cost (\$/kg) by pathway and output dataframe.
output_folder : str, default 'outputs'
    Path for saving output .csv files.
    
Returns
-------
df_output
    Return dataframe containing input parameters and results (energy consumption, costs, etc.). 
LOHC_tot_H2_cost_usd_per_kg
    Return total \\$/kg H2 costs for LOHC delivery pathway ("LOHC"). 
GH2_tot_H2_cost_usd_per_kg
    Return total \\$/kg H2 costs for compressed gaseous hydrogen delivery pathway ("GH2").
LH2_tot_H2_cost_usd_per_kg
    Return total \\$/kg H2 costs for liquid hydrogen delivery pathway ("LH2").

---

