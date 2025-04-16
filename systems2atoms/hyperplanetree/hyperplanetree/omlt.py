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
        fm = lt_regressor.linear_combinations_transform.final_matrix

        used_cols = list(range(len(input_bounds_matrix)))
        summary = copy.deepcopy(lt_regressor.summary())
        for node in summary.values():
            # If node has a 'col', append it to the list of used columns if it's not already there
            if 'col' in node and node['col'] not in used_cols:
                used_cols.append(node['col'])

        used_cols = sorted(used_cols)

        # Remove unused columns from the final matrix
        fm = fm[:, used_cols]

        input_bounds = {}
        for i, row in enumerate(fm.T):
            if i < len(input_bounds_matrix):
                input_bounds[i] = tuple(input_bounds_matrix[i].tolist())
            else:
                # Find extremes the linear combination can be based on the input_bounds
                min_bound = 0
                max_bound = 0
                for j, val in enumerate(row):
                    if val > 0:
                        min_bound += val * input_bounds_matrix[j][0]
                        max_bound += val * input_bounds_matrix[j][1]
                    else:
                        min_bound += val * input_bounds_matrix[j][1]
                        max_bound += val * input_bounds_matrix[j][0]
                
                input_bounds[i] = (min_bound.item(), max_bound.item())

        # Update the cols in the summary to match the new matrix
        for node in summary.values():
            if 'col' in node:
                node['col'] = used_cols.index(node['col'])

            if isinstance(node['models'], TorchLinearRegression):
                # Convert to list and add zeros for all linear combinations features
                zeros_to_add = torch.zeros(len(fm.T) - len(input_bounds_matrix), node['models'].params.shape[1])
                node['models'].params = torch.cat((node['models'].params, zeros_to_add))

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
