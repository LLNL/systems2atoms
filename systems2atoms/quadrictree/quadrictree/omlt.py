import copy
import numpy as np
import torch

try:
    from pyomo.environ import ConstraintList
except ImportError:
    raise ImportError('To use HPTs in optimization, please install Pyomo: https://www.pyomo.org')

try:
    from omlt.linear_tree import LinearTreeDefinition, LinearTreeGDPFormulation, LinearTreeHybridBigMFormulation
except ImportError:
    raise ImportError('To use HPTs in optimization, please install a recent version of OMLT: https://github.com/cog-imperial/OMLT')

from ._classes import TorchLinearRegression

class HyperplaneTreeDefinition(LinearTreeDefinition):
    """OMLT Definition for Hyperplane Trees """
    def __init__(
        self,
        lt_regressor,
        input_bounds_matrix = None,
        scaling_object = None,
    ):
        # Must set up bounds for all linear combinations features (OMLT limitation)
        # Currently, these bounds are not very tight
        # This should get handled by the solver pre-processing anyway
        fm = lt_regressor.linear_combinations_transform.final_matrix
        max_bound = torch.max(torch.abs(torch.abs(input_bounds_matrix.T @ fm))).item()

        input_bounds = {}
        for i, row in enumerate(fm.T):
            if i < len(input_bounds_matrix):
                input_bounds[i] = tuple(input_bounds_matrix[i].tolist())
            else:
                input_bounds[i] = (-max_bound, max_bound)

        summary = copy.deepcopy(lt_regressor.summary())
        for node in summary.values():
            if isinstance(node['models'], TorchLinearRegression):
                # Convert to list and add zeros for all linear combinations features
                node['models'].params = node['models'].params.tolist() + list(np.zeros(len(input_bounds) - len(input_bounds_matrix)))

        super().__init__(
            lt_regressor = summary,
            unscaled_input_bounds = input_bounds,
            scaling_object = scaling_object,
            )
        
        self.fm = fm.numpy()
        
    @property
    def n_input(self):
        return len(self.fm)
        

class HyperplaneTreeOMLTFormulationMixin():
    """A Mixin for OMLT linear tree formulations for Hyperplane Trees"""
    def _build_formulation(self):
        super()._build_formulation()

        self.block.hyperplane_constraints = ConstraintList()

        I = range(len(self.model_definition.fm[0])) # Columns, tree input vars
        J = range(len(self.model_definition.fm)) # Rows, block input vars

        for i in I:
            self.block.hyperplane_constraints.add(sum(self.model_definition.fm[j,i]*self.block.inputs[j] for j in J) == self.block.inputs[i])

class HyperplaneTreeGDPFormulation(HyperplaneTreeOMLTFormulationMixin, LinearTreeGDPFormulation):
    """GDP formulation"""
    pass

class HyperplaneTreeHybridBigMFormulation(HyperplaneTreeOMLTFormulationMixin, LinearTreeHybridBigMFormulation):
    """MIQCP formulation"""
    pass