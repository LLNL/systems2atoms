# hyperplanetree
A Python library to build piecewise linear or piecewise quadratic models in multi-dimensional spaces.

This code is a fork of [linear-tree](https://github.com/cerlymarco/linear-tree). Please see this repository for more information about Linear Model Decision Trees!

## What does this fork include?

The main features of this fork (compared to upstream) are as follows:

1. Translate the mathematics of linear-tree into PyTorch tensor operations. This is roughly 1.5-2.5x faster on CPU and enables GPU calculations. (The exact speedup is highly dependent on your dataset and model hyperparameter choices.)
2. HyperplaneTree: Hyperplanes (linear combinations of features) are considered as splitting variables. This significantly increases the training cost of the tree, which motivated the PyTorch rewrite.
3. QuadricTree: Quadrics (linear combinations of bilinear features) are considered as splitting variables, and the bilinear terms are used in the leaf regressions. Also carries significant cost increase.
4. "Mixed-integer linear program" (MIP) formulations for hyperplane trees via [OMLT](https://github.com/cog-imperial/OMLT) and [Pyomo](https://pyomo.org).

## Why Hyperplanes/Quadrics?
TLDR: Expanding the search space of possible splits can allow us to build trees with better accuracy for the same number of leaves.

We use leaf-model decision trees as surrogates in optimization problems. See: [Ammari et al. Linear model decision trees as surrogates in optimization of engineering applications](https://www.sciencedirect.com/science/article/pii/S009813542300217X)

Expanding the search space of possible splits can allow us to build trees with better accuracy for the same number of leaves. This is useful because when we translate the trees to optimization problems (via OMLT and Pyomo), each leaf becomes a binary variable. Optimization problems generally have poor scaling with the number of binary variables, so we cannot endlessly deepen our trees to achieve high accuracy.

Hyperplanes are specifically useful because they are linear. When converted to optimization constraints, the problem will still be linear: a Mixed-Integer Linear Problem (MIP). 

To further increase the accuracy of our models, we can also use quadratic terms in the splits and leaf regressions. This turns the problem into a Mixed-Integer Quadratic Constrained Problem (MIQCP).

Further increases in model complexity are possible. However, they are not implemented here as they quickly become computationally impractical even for small regression tasks.

## Roadmap
☑ Completed

☐ Planned

☒ Not currently planned

### <ins>Featurizations</ins>
☑ Linear

☑ Hyperplane

☑ Quadric

### <ins>Algorithms</ins>
☑ Tree

☐ Forest

☒ Boost

### <ins>Estimators</ins>
☑ Regressor

☐ Classifier

### <ins>OMLT Formulations</ins>
☑ LinearTreeRegressor (already in OMLT)

☑ HyperplaneTreeRegressor

☐ QuadricTreeRegressor

☐ {Linear,Hyperplane,Quadric}TreeClassifier

☒ {Linear,Hyperplane,Quadric}{Forest,Boost}{Regressor,Classifier}
