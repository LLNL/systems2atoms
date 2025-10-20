# hyperplanetree
A Python library to build piecewise linear models in multi-dimensional spaces.

This code is a fork of [linear-tree](https://github.com/cerlymarco/linear-tree). Please see this repository for more information about Linear Model Decision Trees!

[Check out our paper on this project!](https://www.sciencedirect.com/science/article/pii/S009813542500208X)

## What does this fork include?

The main features of this fork (compared to upstream) are as follows:

1. Translate the mathematics of linear-tree into PyTorch tensor operations. This is roughly 1.5-2.5x faster on CPU and enables GPU calculations. (The exact speedup is highly dependent on your dataset, model hyperparameter choices, and probably even your CPU.)
2. HyperplaneTree: Hyperplanes (linear combinations of features) are considered as splitting variables. This significantly increases the training cost of the tree, which motivated the PyTorch rewrite.
3. "Mixed-integer linear program" (MIP) formulations for hyperplane trees via [OMLT](https://github.com/cog-imperial/OMLT) and [Pyomo](https://pyomo.org).

## Installation
This package can be installed as part of systems2atoms:

```pip install git+https://github.com/LLNL/systems2atoms```

Then, it can be imported as follows:

```python
from systems2atoms.hyperplanetree import LinearTreeRegressor, HyperplaneTreeRegressor
```

Alternatively, you can install hyperplanetree without installing the rest of systems2atoms:

```pip install "hyperplanetree @ git+https://git@github.com/LLNL/systems2atoms#subdirectory=systems2atoms/hyperplanetree"```

The import will then be:

```python
from hyperplanetree import LinearTreeRegressor, HyperplaneTreeRegressor
```

## Quickstart
```python
from systems2atoms.hyperplanetree import LinearTreeRegressor, HyperplaneTreeRegressor
model = HyperplaneTreeRegressor()
model.fit(X, y)
```
Note that `X` and `y` must be PyTorch tensors. Other than that, the `LinearTreeRegressor`/`HyperplaneTreeRegressor` objects behave much like any other sk-learn model.

Please see the notebooks folder for basic tutorials on these models.

## Why Hyperplanes?
TLDR: Expanding the search space of possible splits can allow us to build trees with better accuracy for the same number of leaves.

We use leaf-model decision trees as surrogates in optimization problems. See: [Ammari et al. Linear model decision trees as surrogates in optimization of engineering applications](https://www.sciencedirect.com/science/article/pii/S009813542300217X)

Expanding the search space of possible splits can allow us to build trees with better accuracy for the same number of leaves. This is useful because when we translate the trees to optimization problems (via OMLT and Pyomo), each leaf becomes a binary variable. Optimization problems generally have poor scaling with the number of binary variables, so we cannot endlessly deepen our trees to achieve high accuracy.

Hyperplanes are specifically useful because they are linear. When converted to optimization constraints, the problem will still be linear: a Mixed-Integer Linear Problem (MIP). 

To further increase the accuracy of our models, we can also use quadratic terms in the splits and leaf regressions. This turns the problem into a Mixed-Integer Quadratic Constrained Problem (MIQCP). Quadric Trees are implemented as an experimental feature in this repository. We have not yet seen enough error metric improvement from these models to justify their use.

Further increases in model complexity are possible. However, they are not implemented here as they quickly become computationally impractical even for small regression tasks.
