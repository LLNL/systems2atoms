# Systems2Atoms Surrogates

Examples of surrogate models for cross-scale optimization.

Required software:
- linear-tree \[[original source](https://github.com/cerlymarco/linear-tree)\] \[[emsunshine fork](https://github.com/emsunshine/linear-tree)\] - This is the implementation of linear model decision trees used in this project. The emsunshine fork is recommended for a few additional features, such as the ability to save and load trees.
- [scikit-learn](https://scikit-learn.org/stable/) - A requirement of linear-tree.
- [Pyomo](http://www.pyomo.org/) - Modeling and optimization framework in Python.
- [OMLT](https://github.com/cog-imperial/OMLT/tree/main) - Used for importing linear-tree models into Pyomo.
- A mixed-integer linear program solver, such as [GLPK](https://www.gnu.org/software/glpk/) (FOSS) or [Gurobi](https://www.gurobi.com/) (license required).
