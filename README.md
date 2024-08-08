# systems2atoms

systems2atoms is a suite of models that can simulate the performance of hydrogen
infrastructure across multiple scales.  Examples of the different scale levels include:

1. System Scale - techno-economic analysis for hydrogen applications,
integration of hydrogen infrastructure with the electrical grid,
and supply and demand constraints across applications

2. Component Scale - continuous-flow catalyst beds, electrochemical conversion
units, gas-phase separators, pumps, compressors, storage tanks, evaporators,
heaters, and coolers

3. Material Scale - surface reactions, degradation mechanisms, and catalyst
poisoning

The scope of the software is limited to hydrogen in its most abundant isotopic
form, as protium. Further, it is limited to the use of hydrogen (protium) as a
fuel; that is, as an energy carrier capable of producing energy via exothermic
reactions. This includes: production, storage, distribution, and use as a fuel;
the development of related technologies; and compounds and mixtures in gaseous,
liquid, and solid states.

## Requirements, Installation, and Quickstart

Python 3.8 or higher is recommended.

Individual submodules (e.g. `s2a.surrogates`, `s2a.systems`) have associated requirements. Please
check the READMEs for the individual submodules to find the requirements.

Install this package:
```
pip install git+https://github.com/LLNL/systems2atoms
```

Quick demo of surrogate model optimization (requires [`s2a.surrogates` requirements](analyses/surrogates/README.md)):
```python
import systems2atoms as s2a
model = s2a.surrogates.model_initializer()
s2a.surrogates.solve(model, solver='glpk')
s2a.surrogates.pprint(model)
```

Quick demo of hydrogen delivery cost calculations:
```python
import systems2atoms as s2a
print(s2a.systems.calcs())
```


## Contributing

Please submit any bugfixes or feature improvements as [pull requests](https://help.github.com/articles/using-pull-requests/).


## Authors

* Sneha Akhade
* Matthew McNenly


## License

systems2atoms is distributed under the terms of the MIT license. All new contributions must be under this license.

See LICENSE and NOTICE for details.

SPDX-License-Identifier: MIT

LLNL-CODE-856566


## Acknowledgements

The development of systems2atoms is supported by the Laboratory Directed Research and Development (LDRD) program 
at Lawrence Livermore National Laboratory (LLNL). The project identifier is `GS 23-ERD-016` with Sneha Akhade as
principal investigator.
