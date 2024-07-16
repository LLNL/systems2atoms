# Systems2Atoms Surrogates

Examples of surrogate models for cross-scale optimization.

## Requirements

This submodule requires the following software:
- linear-tree \[[original source](https://github.com/cerlymarco/linear-tree)\] \[[emsunshine fork](https://github.com/emsunshine/linear-tree)\] - This is the implementation of linear model decision trees used in this project. The emsunshine fork is recommended for a few additional features, such as the ability to save and load trees.
- [scikit-learn](https://scikit-learn.org/stable/) - A requirement of linear-tree.
- [TQDM](https://github.com/tqdm/tqdm) Progress bars in Python.
- [Pyomo](http://www.pyomo.org/) - Modeling and optimization framework in Python.
- [OMLT](https://github.com/cog-imperial/OMLT/tree/main) - Used for importing linear-tree models into Pyomo. Need versions from 2024 or newer for linear-tree support.
- A mixed-integer linear program solver, such as [GLPK](https://www.gnu.org/software/glpk/) (FOSS) or [Gurobi](https://www.gurobi.com/) (license required).

The python requirements can be installed from github:
```
pip install git+https://github.com/cog-imperial/OMLT
pip install git+https://github.com/emsunshine/linear-tree
```
These packages should automatically install scikit-learn and Pyomo as dependencies if needed.

Do not forget to install a mixed-integer linear program solver!
