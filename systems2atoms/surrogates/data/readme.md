# About this data

This data was obtained computationally using the code in the `systems` module found in this repository. 

# Units

## Cost Data
### (Source: System-scale model)
| Column | Name | Unit |
|--------|------|------|
| 0 | Station Capacity | kg $H_2$ per day |
| 1 | CSTR Volume | $m^3$ |
| 2 | Hydrogen Yield | Fraction |
| 3 | Catalyst Mass | kg |
| 4 | Temperature | K |
| 5 | Pressure | bar |
| 6 | Catalyst Price | $ per kg metal |
| 7 | Leveilzed Cost of Hydrogen | $ per kg $H_2$


## Component Data
### (Source: Component-scale model)
| Column | Name | Unit |
|--------|------|------|
| 0 | CSTR Volume | $m^3$ |
| 1 | Pellet Effectivenes Factor | Fraction |
| 2 | Dehydrogenation Reaction Order | Unitless |
| 3 | Turnover Frequency | Hz |
| 4 | Catalyst Metal Amount | mol |
| 5 | Hydrogen Yield | Fraction |

## Nanoparticle Data
### (Source: Material-scale model)
| Column | Name | Unit |
|--------|------|------|
| 0 | Catalyst Metal Choice | N/A |
| 1 | Temperature | K |
| 2 | Pressure | bar |
| 3 | Pressure | atm |
| 4 | Turnover Frequency | (mol H2/mol Pd*hr) |
| 5 | Turnover Frequency | (mol H2/mol Pd*sec) |
